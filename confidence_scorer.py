import bisect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


DAY_TO_SPARK_DAYOFWEEK = {
    "sunday": 1,
    "monday": 2,
    "tuesday": 3,
    "wednesday": 4,
    "thursday": 5,
    "friday": 6,
    "saturday": 7,
}


RESIDUAL_FALLBACK_LEVELS: Sequence[Tuple[str, Tuple[str, ...]]] = (
    ("segment_daytype_bucket", ("from_bpuic", "to_bpuic", "day_type", "time_bucket_15min")),
    ("segment_hour_of_week", ("from_bpuic", "to_bpuic", "hour_of_week")),
    ("segment_day_of_week", ("from_bpuic", "to_bpuic", "day_of_week")),
    ("segment_day_type", ("from_bpuic", "to_bpuic", "day_type")),
    ("segment", ("from_bpuic", "to_bpuic")),
    ("global_daytype_bucket", ("day_type", "time_bucket_15min")),
    ("global", ()),
)


def seconds_to_bucket(seconds: int) -> int:
    """Convert seconds from midnight to a 15-minute bucket index, 0..95."""
    seconds = int(seconds) % (24 * 3600)
    return int((seconds // 60) // 15)


def event_hour(seconds: int) -> int:
    """Convert seconds from midnight to hour of day, 0..23."""
    return int((int(seconds) % (24 * 3600)) // 3600)


def day_type_from_day_of_week(day_of_week: int) -> str:
    """Return weekday/weekend using Spark's day-of-week convention."""
    return "weekend" if int(day_of_week) in (1, 7) else "weekday"


def hour_of_week(day_of_week: int, hour: int) -> int:
    """Return the notebook's hour-of-week bucket using Spark day-of-week."""
    return (int(day_of_week) - 1) * 24 + int(hour)


def bpuic_from_stop_id(stop_id: Any) -> Optional[int]:
    """
    Extract BPUIC from a GTFS stop id.

    Examples:
        "8591818"     -> 8591818
        "8591818:0:1" -> 8591818
    """
    if stop_id is None:
        return None

    try:
        return int(str(stop_id).split(":")[0])
    except ValueError:
        return None


def arrival_segment_from_bpuic(step: Mapping[str, Any]) -> Optional[int]:
    """Return the adjacent segment start BPUIC for the arrival into step['to']."""
    return bpuic_from_stop_id(step.get("arrival_segment_from_stop", step.get("from")))


def _norm(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


class PandasResidualCDFScorer:
    """
    Score route events from the residual CDF lookup written by the notebook.

    The notebook trains the reduced delay model and builds validation residual
    distributions. For one event:

        margin_min = usable_slack_min - predicted_delay_min
        P(success) = P(residual_min <= margin_min)

    Expected lookup_df columns:
        level
        residual_threshold_min
        success_probability
        n_obs

    Plus the key columns used by each fallback level, such as from_bpuic,
    to_bpuic, day_type, time_bucket_15min, hour_of_week, and day_of_week.
    """

    def __init__(
        self,
        lookup_df: pd.DataFrame,
        default_probability: float = 0.50,
        fallback_levels: Sequence[Tuple[str, Tuple[str, ...]]] = RESIDUAL_FALLBACK_LEVELS,
    ):
        self.lookup_df = lookup_df.copy()
        self.default_probability = float(default_probability)
        self.fallback_levels = tuple(fallback_levels)
        self.index: Dict[Tuple[Any, ...], List[Tuple[float, float, int]]] = {}

        self._validate_columns()
        self._build_index()

    def _validate_columns(self) -> None:
        if "success_probability" not in self.lookup_df.columns and "cdf_probability" in self.lookup_df.columns:
            self.lookup_df = self.lookup_df.rename(columns={"cdf_probability": "success_probability"})

        required = {
            "level",
            "residual_threshold_min",
            "success_probability",
            "n_obs",
        }

        missing = required - set(self.lookup_df.columns)

        if missing:
            raise ValueError(
                f"lookup_df is missing required columns: {sorted(missing)}. "
                f"Current columns are: {list(self.lookup_df.columns)}"
            )

    def _row_key(self, level: str, row: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
        for known_level, key_cols in self.fallback_levels:
            if known_level != level:
                continue

            values = tuple(_norm(row.get(c)) for c in key_cols)

            if any(v is None for v in values):
                return None

            return (level, *values)

        return None

    def _event_key(self, level: str, event: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
        for known_level, key_cols in self.fallback_levels:
            if known_level != level:
                continue

            values = tuple(_norm(event.get(c)) for c in key_cols)

            if any(v is None for v in values):
                return None

            return (level, *values)

        return None

    def _build_index(self) -> None:
        for _, row in self.lookup_df.iterrows():
            level = str(row["level"])
            key = self._row_key(level, row)

            if key is None:
                continue

            self.index.setdefault(key, []).append(
                (
                    float(row["residual_threshold_min"]),
                    float(row["success_probability"]),
                    int(row["n_obs"]) if not pd.isna(row["n_obs"]) else 0,
                )
            )

        for values in self.index.values():
            values.sort(key=lambda x: x[0])

    @staticmethod
    def _interp(values: List[Tuple[float, float, int]], margin_min: float) -> float:
        """Linearly interpolate the CDF value between residual thresholds."""
        xs = [v[0] for v in values]
        i = bisect.bisect_left(xs, float(margin_min))

        if i <= 0:
            return values[0][1]

        if i >= len(values):
            return values[-1][1]

        x0, y0, _ = values[i - 1]
        x1, y1, _ = values[i]

        if x1 == x0:
            return y1

        weight = (float(margin_min) - x0) / (x1 - x0)
        return float(y0 + weight * (y1 - y0))

    @staticmethod
    def _predicted_delay_min(event: Mapping[str, Any]) -> float:
        if "predicted_delay_min" in event:
            return float(event["predicted_delay_min"])

        if "prediction" in event:
            return float(event["prediction"])

        raise ValueError(
            "Each event must include predicted_delay_min or prediction before "
            "residual-CDF scoring. Use the reduced model from the notebook to "
            "add that value."
        )

    @staticmethod
    def _usable_slack_min(event: Mapping[str, Any]) -> float:
        if "usable_slack_min" in event:
            return float(event["usable_slack_min"])

        return float(event.get("slack_seconds", 0.0)) / 60.0

    def probability_for_event(self, event: Mapping[str, Any]) -> float:
        """
        Return P(success) for one transfer/final-arrival event.

        The event must already contain a reduced-model prediction. This function
        only applies the residual lookup and fallback hierarchy.
        """
        predicted_delay_min = self._predicted_delay_min(event)
        usable_slack_min = self._usable_slack_min(event)
        margin_min = usable_slack_min - predicted_delay_min

        for level, _ in self.fallback_levels:
            key = self._event_key(level, event)

            if key is None:
                continue

            values = self.index.get(key)

            if values:
                p = self._interp(values, margin_min)
                return max(0.0, min(1.0, float(p)))

        return self.default_probability


def extract_probability_events_from_route(
    route: Mapping[str, Any],
    day: str,
    min_transfer_seconds: int = 120,
) -> List[Dict[str, Any]]:
    """
    Convert one route from RobustJourneyPlanner into segment events.

    These events still need model predictions before they can be scored:
    add predicted_delay_min with the reduced delay model, then pass them to
    PandasResidualCDFScorer.
    """
    day_key = str(day).lower()

    if day_key not in DAY_TO_SPARK_DAYOFWEEK:
        raise ValueError(f"Unknown day: {day}")

    day_of_week = DAY_TO_SPARK_DAYOFWEEK[day_key]
    day_type = day_type_from_day_of_week(day_of_week)
    steps = list(route.get("steps", []))
    events: List[Dict[str, Any]] = []

    for i in range(len(steps) - 1):
        current = steps[i]
        next_step = steps[i + 1]

        if current.get("type") != "TRANSIT":
            continue

        if i == len(steps) - 2 and next_step.get("type") == "WALK":
            continue

        scheduled_arrival = int(current["arr_time"])
        hour = event_hour(scheduled_arrival)

        events.append(
            {
                "event_type": "transfer",
                "trip_id": current.get("trip_id"),
                "from_bpuic": arrival_segment_from_bpuic(current),
                "to_bpuic": bpuic_from_stop_id(current.get("to")),
                "bpuic": bpuic_from_stop_id(current.get("to")),
                "scheduled_arrival": scheduled_arrival,
                "day_of_week": day_of_week,
                "day_type": day_type,
                "hour": hour,
                "hour_of_week": hour_of_week(day_of_week, hour),
                "time_bucket_15min": seconds_to_bucket(scheduled_arrival),
                "slack_seconds": int(
                    next_step["dep_time"]
                    - scheduled_arrival
                    - min_transfer_seconds
                ),
            }
        )

    previous_transit = None

    for step in reversed(steps):
        if step.get("type") == "TRANSIT":
            previous_transit = step
            break

    if previous_transit is not None:
        scheduled_arrival = int(previous_transit["arr_time"])
        hour = event_hour(scheduled_arrival)

        if steps and steps[-1].get("type") == "WALK":
            scheduled_route_arrival = int(steps[-1]["arr_time"])
        else:
            scheduled_route_arrival = scheduled_arrival

        slack = int(route["arrival_deadline_seconds"] - scheduled_route_arrival)

        events.append(
            {
                "event_type": "final_arrival",
                "trip_id": previous_transit.get("trip_id"),
                "from_bpuic": arrival_segment_from_bpuic(previous_transit),
                "to_bpuic": bpuic_from_stop_id(previous_transit.get("to")),
                "bpuic": bpuic_from_stop_id(previous_transit.get("to")),
                "scheduled_arrival": scheduled_arrival,
                "scheduled_route_arrival": scheduled_route_arrival,
                "day_of_week": day_of_week,
                "day_type": day_type,
                "hour": hour,
                "hour_of_week": hour_of_week(day_of_week, hour),
                "time_bucket_15min": seconds_to_bucket(scheduled_arrival),
                "slack_seconds": int(slack),
            }
        )

    for event in events:
        event["usable_slack_min"] = float(event["slack_seconds"]) / 60.0

    return events


def score_routes_with_residual_cdf(
    routes: Sequence[Mapping[str, Any]],
    scorer: PandasResidualCDFScorer,
    day: str,
    predict_delay_min: Callable[[Mapping[str, Any]], float],
    min_confidence: float = 0.0,
    min_transfer_seconds: int = 120,
) -> List[Dict[str, Any]]:
    """
    Score routes by multiplying event probabilities.

    predict_delay_min receives one event dict and must return the reduced-model
    delay prediction in minutes. In the cluster notebook this can be backed by a
    Spark batch prediction; for small demos it can be a local lookup.
    """
    scored: List[Dict[str, Any]] = []

    for route in routes:
        confidence = 1.0

        events = extract_probability_events_from_route(
            route,
            day=day,
            min_transfer_seconds=min_transfer_seconds,
        )

        event_scores = []

        for event in events:
            event = dict(event)
            event["predicted_delay_min"] = float(predict_delay_min(event))
            p = scorer.probability_for_event(event)
            confidence *= p

            event_scores.append(
                {
                    **event,
                    "success_probability": float(p),
                }
            )

        new_route = dict(route)
        new_route["confidence"] = float(confidence)
        new_route["probability_events"] = event_scores

        if confidence >= min_confidence:
            scored.append(new_route)

    return sorted(
        scored,
        key=lambda r: (
            -int(r.get("departure_seconds", 0)),
            -float(r.get("confidence", 0.0)),
            float(r.get("walk_m", 0.0)),
        ),
    )
