"""
End-to-end robust journey planning helpers and interactive dashboard.

This module links:
1. Timetable candidate routes from journey_planner.py
2. Reduced delay-model predictions
3. Validation residual CDF probabilities
4. A beautiful, feature-rich interactive Jupyter widgets dashboard
"""

import math
import os
import bisect
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
from contextlib import closing

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, StringType

from journey_planner import RobustJourneyPlanner

# --- Core Definitions ---

from robust_udfs import RESIDUAL_GRID_MIN

RESIDUAL_LEVEL_KEYS: Mapping[str, Tuple[str, ...]] = {
    "segment_daytype_bucket": ("from_bpuic", "to_bpuic", "day_type", "time_bucket_15min"),
    "segment_hour_of_week": ("from_bpuic", "to_bpuic", "hour_of_week"),
    "segment_day_of_week": ("from_bpuic", "to_bpuic", "day_of_week"),
    "segment_day_type": ("from_bpuic", "to_bpuic", "day_type"),
    "segment": ("from_bpuic", "to_bpuic"),
    "global_daytype_bucket": ("day_type", "time_bucket_15min"),
    "global": (),
}

# Global references surviving module reloads by attaching them to sys module
import sys
if not hasattr(sys, "_PRETRIP_HISTORY_CACHE"):
    sys._PRETRIP_HISTORY_CACHE = {}
if not hasattr(sys, "_PLANNER_CACHE"):
    sys._PLANNER_CACHE = {}
if not hasattr(sys, "_ALL_STOP_NAMES_CACHE"):
    sys._ALL_STOP_NAMES_CACHE = {}

def normalize_mode_value(value: Any) -> str:
    value = "UNKNOWN" if pd.isna(value) else str(value).strip().upper()
    if value in {"", "?", "NULL", "NAN"}:
        return "UNKNOWN"
    if value in {"B", "BUS", "BUSS"}:
        return "BUS"
    if value in {"T", "TRAM"}:
        return "TRAM"
    if value in {"M", "METRO"}:
        return "METRO"
    if value in {"BAT", "BOAT", "SHIP", "SCHIFF"}:
        return "SHIP"
    if value in {"ZAHNRADBAHN", "CC"}:
        return "RACK_RAIL"
    return value


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    radius_m = 6_371_000.0
    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius_m * math.asin(math.sqrt(a))


def build_walk_edges_from_stops(stops_pd: pd.DataFrame, max_walk_m: float = 500.0) -> pd.DataFrame:
    cols = ["a_stop_id", "b_stop_id", "distance"]
    if stops_pd.empty:
        return pd.DataFrame(columns=cols)

    base = stops_pd.dropna(subset=["stop_lat", "stop_lon"]).drop_duplicates("stop_id").copy()
    rows = []
    values = base[["stop_id", "stop_lat", "stop_lon"]].to_records(index=False)
    for i in range(len(values)):
        stop_i, lat_i, lon_i = values[i]
        for j in range(i + 1, len(values)):
            stop_j, lat_j, lon_j = values[j]
            distance = haversine_m(lat_i, lon_i, lat_j, lon_j)
            if 0 < distance <= max_walk_m:
                rows.append((str(stop_i), str(stop_j), float(distance)))
    return pd.DataFrame(rows, columns=cols)


def load_planner_inputs_from_trino(
    conn,
    cfg,
    max_walk_m: float = 500.0,
    day: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load planner inputs from Trino.
    The timetable is filtered to one day when `day` is provided.
    """
    day_filter = ""
    if day is not None:
        day_col = str(day).lower()
        if day_col not in {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}:
            raise ValueError(f"Unknown day: {day}")
        day_filter = f" AND c.{day_col} = TRUE"

    sharedNS = "iceberg.com490_iceberg"
    username = cfg.username
    namespace = f"iceberg.{username}"
    region_table = f"{namespace}.region_stops_{cfg.region_name}"

    stop_times_sql = f"""
    WITH trips_touching_region AS (
        SELECT DISTINCT st.trip_id
        FROM {sharedNS}.sbb_stop_times st
        JOIN {region_table} rs
          ON st.stop_id = rs.stop_id
        WHERE st.pub_date = DATE '{cfg.timetable_date}'
    )
    SELECT
        st.trip_id,
        st.stop_id,
        st.stop_sequence,
        st.arrival_time,
        st.departure_time,
        c.monday, c.tuesday, c.wednesday, c.thursday, c.friday, c.saturday, c.sunday,
        COALESCE(CAST(r.agency_id AS varchar), 'UNKNOWN') AS operator_id,
        COALESCE(CAST(r.route_id AS varchar), CAST(t.route_id AS varchar), 'UNKNOWN') AS line_id,
        COALESCE(NULLIF(CAST(r.route_short_name AS varchar), ''), NULLIF(CAST(t.trip_short_name AS varchar), ''), 'UNKNOWN') AS line_text,
        COALESCE(NULLIF(CAST(r.route_desc AS varchar), ''), 'UNKNOWN') AS transport_clean,
        COALESCE(NULLIF(CAST(r.route_desc AS varchar), ''), 'UNKNOWN') AS product_id_clean
    FROM {sharedNS}.sbb_stop_times st
    JOIN trips_touching_region tr
      ON st.trip_id = tr.trip_id
    JOIN {sharedNS}.sbb_trips t
      ON st.pub_date = t.pub_date AND st.trip_id = t.trip_id
    JOIN {sharedNS}.sbb_calendar c
      ON t.pub_date = c.pub_date AND t.service_id = c.service_id
    LEFT JOIN {sharedNS}.sbb_routes r
      ON t.pub_date = r.pub_date AND t.route_id = r.route_id
    WHERE st.pub_date = DATE '{cfg.timetable_date}'
      {day_filter}
    ORDER BY st.trip_id, st.stop_sequence
    """
    with closing(conn.cursor()) as cursor:
        cursor.execute(stop_times_sql)
        stop_times_pd = pd.DataFrame(cursor.fetchall(), columns=[c[0] for c in cursor.description])

    if stop_times_pd.empty:
        raise ValueError("No timetable rows found for the configured region/date.")

    stop_times_pd["stop_id"] = stop_times_pd["stop_id"].astype(str)
    for col in ["operator_id", "line_id", "line_text"]:
        stop_times_pd[col] = stop_times_pd[col].fillna("UNKNOWN").astype(str)
    for col in ["transport_clean", "product_id_clean"]:
        stop_times_pd[col] = stop_times_pd[col].apply(normalize_mode_value)
    for col in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        stop_times_pd[col] = stop_times_pd[col].astype(bool)

    stops_sql = f"""
    WITH route_stops AS (
        SELECT DISTINCT st.stop_id
        FROM {sharedNS}.sbb_stop_times st
        JOIN (
            SELECT DISTINCT st2.trip_id
            FROM {sharedNS}.sbb_stop_times st2
            JOIN {region_table} rs
              ON st2.stop_id = rs.stop_id
            WHERE st2.pub_date = DATE '{cfg.timetable_date}'
        ) tr ON st.trip_id = tr.trip_id
        WHERE st.pub_date = DATE '{cfg.timetable_date}'
    )
    SELECT s.stop_id, s.stop_name, s.stop_lat, s.stop_lon
    FROM {sharedNS}.sbb_stops s
    JOIN route_stops rs
      ON s.stop_id = rs.stop_id
    WHERE s.pub_date = DATE '{cfg.timetable_date}'
      AND NOT s.stop_id LIKE 'Parent%'
    """
    with closing(conn.cursor()) as cursor:
        cursor.execute(stops_sql)
        stops_pd = pd.DataFrame(cursor.fetchall(), columns=[c[0] for c in cursor.description])
    stops_pd["stop_id"] = stops_pd["stop_id"].astype(str)

    with closing(conn.cursor()) as cursor:
        cursor.execute(f"SELECT stop_id, stop_name, stop_lat, stop_lon FROM {region_table}")
        region_stops_for_walk_pd = pd.DataFrame(cursor.fetchall(), columns=[c[0] for c in cursor.description])

    walk_stop_pd = (
        pd.concat([stops_pd, region_stops_for_walk_pd], ignore_index=True)
        .drop_duplicates("stop_id")
    )
    walk_pd = build_walk_edges_from_stops(walk_stop_pd, max_walk_m=max_walk_m)
    print("Planner stops:", len(stops_pd), "stop_times:", len(stop_times_pd), "walk edges:", len(walk_pd))
    return stops_pd, walk_pd, stop_times_pd


def routes_to_event_rows(routes: Sequence[Mapping[str, Any]], service_date: str) -> pd.DataFrame:
    rows = []
    service_day = pd.Timestamp(service_date)
    for route_id, route in enumerate(routes):
        for event_id, event in enumerate(route.get("events_for_model", [])):
            row = dict(event)
            row["route_id"] = int(route_id)
            row["event_id"] = int(event_id)
            row["operating_day"] = service_day.date().isoformat()
            row["scheduled_arrival_ts"] = service_day + pd.to_timedelta(int(row["scheduled_arrival"]), unit="s")
            rows.append(row)
    return pd.DataFrame(rows)


def add_missing_columns(df: DataFrame, defaults: Mapping[str, Tuple[Any, str]]) -> DataFrame:
    for col_name, (default_value, dtype) in defaults.items():
        if col_name not in df.columns:
            df = df.withColumn(col_name, F.lit(default_value).cast(dtype))
    return df


def add_weather_to_route_events(events_df: DataFrame, weather_df: Optional[DataFrame]) -> DataFrame:
    if weather_df is None:
        return events_df
    weather_join_df = (
        weather_df
        .withColumnRenamed("weather_date", "weather_join_date")
        .withColumnRenamed("weather_hour", "weather_join_hour")
    )
    return (
        events_df
        .join(
            weather_join_df,
            (F.col("operating_day") == F.col("weather_join_date")) &
            (F.col("hour") == F.col("weather_join_hour")),
            "left",
        )
        .drop("weather_join_date", "weather_join_hour")
    )


from robust_udfs import nearest_residual_threshold_py
nearest_residual_threshold_udf = F.udf(nearest_residual_threshold_py, DoubleType())


def score_events_with_residual_lookup(events_df: DataFrame, residual_lookup_df: DataFrame) -> DataFrame:
    scored = (
        events_df
        .withColumn("margin_min", F.col("usable_slack_min") - F.col("prediction"))
        .withColumn("residual_threshold_min", nearest_residual_threshold_udf(F.col("margin_min")))
    )

    join_specs = [
        ("segment_daytype_bucket", ["from_bpuic", "to_bpuic", "day_type", "time_bucket_15min"]),
        ("segment_hour_of_week", ["from_bpuic", "to_bpuic", "hour_of_week"]),
        ("segment_day_of_week", ["from_bpuic", "to_bpuic", "day_of_week"]),
        ("segment_day_type", ["from_bpuic", "to_bpuic", "day_type"]),
        ("segment", ["from_bpuic", "to_bpuic"]),
        ("global_daytype_bucket", ["day_type", "time_bucket_15min"]),
        ("global", []),
    ]

    probability_cols = []
    level_cols = []
    for i, (level, keys) in enumerate(join_specs):
        p_col = f"prob_{i}"
        n_col = f"n_obs_{i}"
        level_col = f"level_{i}"
        lookup = (
            residual_lookup_df
            .filter(F.col("level") == level)
            .select(
                *keys,
                "residual_threshold_min",
                F.col("success_probability").alias(p_col),
                F.col("n_obs").alias(n_col),
                F.lit(level).alias(level_col),
            )
        )
        scored = scored.join(lookup, on=keys + ["residual_threshold_min"], how="left")
        probability_cols.append(p_col)
        level_cols.append(level_col)

    scored = scored.withColumn("success_probability", F.coalesce(*[F.col(c) for c in probability_cols], F.lit(0.5)))
    scored = scored.withColumn("residual_level_used", F.coalesce(*[F.col(c) for c in level_cols], F.lit("default")))
    return scored


def score_events_with_residual_lookup_pandas(events_pd: pd.DataFrame, residual_lookup_pd: pd.DataFrame) -> pd.DataFrame:
    events = events_pd.copy()
    events["margin_min"] = events["usable_slack_min"] - events["prediction"]
    events["residual_threshold_min"] = events["margin_min"].apply(nearest_residual_threshold_py)

    join_specs = [
        ("segment_daytype_bucket", ["from_bpuic", "to_bpuic", "day_type", "time_bucket_15min"]),
        ("segment_hour_of_week", ["from_bpuic", "to_bpuic", "hour_of_week"]),
        ("segment_day_of_week", ["from_bpuic", "to_bpuic", "day_of_week"]),
        ("segment_day_type", ["from_bpuic", "to_bpuic", "day_type"]),
        ("segment", ["from_bpuic", "to_bpuic"]),
        ("global_daytype_bucket", ["day_type", "time_bucket_15min"]),
        ("global", []),
    ]

    success_probabilities = []
    levels_used = []

    lookup_by_level = {level: residual_lookup_pd[residual_lookup_pd["level"] == level] for level, _ in join_specs}

    # Cast matching keys to float/int to prevent type mismatches
    for level, keys in join_specs:
        lookup = lookup_by_level[level]
        if lookup.empty:
            continue
        for key in keys:
            if key in ["from_bpuic", "to_bpuic", "time_bucket_15min", "hour_of_week", "day_of_week"]:
                lookup_by_level[level] = lookup.copy()
                lookup_by_level[level][key] = lookup[key].astype(float).fillna(-999).astype(int)

    for _, row in events.iterrows():
        prob = None
        level_used = "default"

        for level, keys in join_specs:
            lookup = lookup_by_level[level]
            if lookup.empty:
                continue

            cond = (lookup["residual_threshold_min"] == float(row["residual_threshold_min"]))
            for key in keys:
                val = row[key]
                if key in ["from_bpuic", "to_bpuic", "time_bucket_15min", "hour_of_week", "day_of_week"]:
                    cond = cond & (lookup[key] == int(float(val)))
                else:
                    cond = cond & (lookup[key] == str(val))

            matched = lookup[cond]
            if not matched.empty:
                prob = float(matched.iloc[0]["success_probability"])
                level_used = level
                break

        if prob is None:
            prob = 0.5

        success_probabilities.append(prob)
        levels_used.append(level_used)

    events["success_probability"] = success_probabilities
    events["residual_level_used"] = levels_used
    return events



def ensure_pretrip_history_lookup_tables(spark, cfg) -> Tuple[Dict[str, DataFrame], Dict[str, float]]:
    if "tables" in sys._PRETRIP_HISTORY_CACHE:
        return sys._PRETRIP_HISTORY_CACHE["tables"], sys._PRETRIP_HISTORY_CACHE["defaults"]

    save_base = cfg.save_base()
    HISTORY_FEATURE_BASE = f"{save_base}/feature_history_reduced"
    HISTORY_LOOKUP_PATHS = {
        "defaults": f"{HISTORY_FEATURE_BASE}/defaults.parquet",
        "segment_hist": f"{HISTORY_FEATURE_BASE}/segment_hist.parquet",
        "segment_daytype_bucket_hist": f"{HISTORY_FEATURE_BASE}/segment_daytype_bucket_hist.parquet",
        "to_stop_daytype_bucket_hist": f"{HISTORY_FEATURE_BASE}/to_stop_daytype_bucket_hist.parquet",
        "operator_hour_hist": f"{HISTORY_FEATURE_BASE}/operator_hour_hist.parquet",
        "line_hour_hist": f"{HISTORY_FEATURE_BASE}/line_hour_hist.parquet",
    }

    defaults = {k: float(v or 0.0) for k, v in spark.read.parquet(HISTORY_LOOKUP_PATHS["defaults"]).first().asDict().items()}
    tables = {
        name: spark.read.parquet(path)
        for name, path in HISTORY_LOOKUP_PATHS.items()
        if name != "defaults"
    }
    sys._PRETRIP_HISTORY_CACHE = {"tables": tables, "defaults": defaults}
    print("LOADED pretrip history lookup tables from:", HISTORY_FEATURE_BASE)
    return tables, defaults


def add_pretrip_history_features_from_lookup(spark, cfg, target_df: DataFrame) -> DataFrame:
    tables, defaults = ensure_pretrip_history_lookup_tables(spark, cfg)
    return (
        target_df
        .join(tables["segment_hist"], on=["from_bpuic", "to_bpuic"], how="left")
        .join(tables["segment_daytype_bucket_hist"], on=["from_bpuic", "to_bpuic", "day_type", "time_bucket_15min"], how="left")
        .join(tables["to_stop_daytype_bucket_hist"], on=["to_bpuic", "day_type", "time_bucket_15min"], how="left")
        .join(tables["operator_hour_hist"], on=["operator_id", "hour"], how="left")
        .join(tables["line_hour_hist"], on=["line_text", "hour"], how="left")
        .fillna({
            "hist_segment_n": 0,
            "hist_segment_avg_delay": defaults["global_avg_delay"],
            "hist_segment_p50_delay": defaults["global_p50_delay"],
            "hist_segment_p90_delay": defaults["global_p90_delay"],
            "hist_segment_p95_delay": defaults["global_p95_delay"],
            "hist_segment_daytype_bucket_avg_delay": defaults["global_avg_delay"],
            "hist_segment_daytype_bucket_p90_delay": defaults["global_p90_delay"],
            "hist_to_stop_daytype_bucket_avg_delay": defaults["global_avg_delay"],
            "hist_operator_hour_avg_delay": defaults["global_avg_delay"],
            "hist_line_hour_avg_delay": defaults["global_avg_delay"],
        })
    )


def prepare_route_events_for_reduced_model(spark, cfg, events_pd: pd.DataFrame, weather_df: Optional[DataFrame]) -> DataFrame:
    if events_pd.empty:
        raise ValueError("No route events to score.")

    events_df = spark.createDataFrame(events_pd)
    events_df = add_missing_columns(events_df, {
        "operator_id": ("UNKNOWN", "string"),
        "line_id": ("UNKNOWN", "string"),
        "line_text": ("UNKNOWN", "string"),
        "transport_clean": ("UNKNOWN", "string"),
        "product_id_clean": ("UNKNOWN", "string"),
        "scheduled_travel_time_min": (0.0, "double"),
        "segment_sequence_proxy": (1, "int"),
        "n_segments_in_trip": (1, "int"),
        "trip_progress": (1.0, "double"),
        "hour": (0, "int"),
        "day_of_week": (2, "int"),
        "time_bucket_15min": (0, "int"),
        "hour_of_week": (24, "int"),
        "day_type": ("weekday", "string"),
    })

    events_df = (
        events_df
        .withColumn("operating_day", F.to_date("operating_day"))
        .withColumn("scheduled_arrival_ts", F.to_timestamp("scheduled_arrival_ts"))
        .withColumn("from_bpuic", F.col("from_bpuic").cast("int"))
        .withColumn("to_bpuic", F.col("to_bpuic").cast("int"))
        .withColumn("from_bpuic_str", F.col("from_bpuic").cast("string"))
        .withColumn("to_bpuic_str", F.col("to_bpuic").cast("string"))
        .withColumn("operator_id", F.coalesce(F.col("operator_id"), F.lit("UNKNOWN")).cast("string"))
        .withColumn("line_text", F.coalesce(F.col("line_text"), F.lit("UNKNOWN")).cast("string"))
        .withColumn("transport_clean", F.coalesce(F.col("transport_clean"), F.lit("UNKNOWN")).cast("string"))
        .withColumn("product_id_clean", F.coalesce(F.col("product_id_clean"), F.lit("UNKNOWN")).cast("string"))
        .withColumn("month", F.month("operating_day"))
        .withColumn("is_weekend", F.col("day_of_week").isin([1, 7]).cast("int"))
        .withColumn("hour_sin", F.sin(2 * F.pi() * F.col("hour") / F.lit(24.0)))
        .withColumn("hour_cos", F.cos(2 * F.pi() * F.col("hour") / F.lit(24.0)))
        .withColumn("how_sin", F.sin(2 * F.pi() * F.col("hour_of_week") / F.lit(168.0)))
        .withColumn("how_cos", F.cos(2 * F.pi() * F.col("hour_of_week") / F.lit(168.0)))
        .withColumn("scheduled_gap_min", F.lit(0.0))
        .withColumn("scheduled_headway_min", F.lit(180.0))
    )

    events_df = add_weather_to_route_events(events_df, weather_df)
    for col_name in ["temperature_2m", "rain_mm", "precipitation_mm", "snowfall_cm", "precipitation_past_3h", "is_daylight"]:
        if col_name not in events_df.columns:
            events_df = events_df.withColumn(col_name, F.lit(0.0))

    events_df = events_df.fillna({
        "temperature_2m": 0.0,
        "rain_mm": 0.0,
        "precipitation_mm": 0.0,
        "snowfall_cm": 0.0,
        "precipitation_past_3h": 0.0,
        "is_daylight": 0,
        "scheduled_travel_time_min": 0.0,
        "segment_sequence_proxy": 1,
        "n_segments_in_trip": 1,
        "trip_progress": 1.0,
    })

    return add_pretrip_history_features_from_lookup(spark, cfg, events_df)


def route_display_signature(route: Mapping[str, Any], planner: RobustJourneyPlanner) -> Tuple:
    signature = []
    for step in route.get("steps", []):
        if step.get("type") != "TRANSIT":
            continue
        signature.append(
            (
                str(step.get("trip_id")),
                planner._name(step.get("from")),
                planner._name(step.get("to")),
            )
        )
    return tuple(signature)


def deduplicate_routes_for_display(
    routes: Sequence[Mapping[str, Any]],
    planner: RobustJourneyPlanner,
) -> List[Dict[str, Any]]:
    sorted_routes = sorted(
        routes,
        key=lambda r: (
            -int(r.get("departure_seconds", 0)),      # teacher requirement
            -float(r.get("confidence", 0.0)),         # keep best variant
            int(r.get("transit_legs", 999)),
            float(r.get("walk_m", 0.0)),
        ),
    )
    deduped = []
    seen = set()
    for route in sorted_routes:
        sig = route_display_signature(route, planner)
        if not sig:
            sig = tuple(
                (step.get("type"), planner._name(step.get("from")), planner._name(step.get("to")))
                for step in route.get("steps", [])
            )
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(dict(route))
    return deduped


def attach_scored_events_to_routes(
    routes: Sequence[Mapping[str, Any]],
    scored_events_input: Any,
    min_confidence: float,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    event_cols = [
        "route_id", "event_id", "event_type", "from_bpuic", "to_bpuic", "scheduled_arrival",
        "day_type", "time_bucket_15min", "hour_of_week", "day_of_week",
        "usable_slack_min", "prediction", "margin_min", "success_probability", "residual_level_used",
        "n_obs_0", "n_obs_1", "n_obs_2", "n_obs_3", "n_obs_4", "n_obs_5", "n_obs_6",
    ]
    if isinstance(scored_events_input, pd.DataFrame):
        event_scores_pd = scored_events_input[[c for c in event_cols if c in scored_events_input.columns]].copy()
    else:
        event_scores_pd = scored_events_input.select(*[c for c in event_cols if c in scored_events_input.columns]).toPandas()
    scored_routes = []
    for route_id, route in enumerate(routes):
        event_scores = event_scores_pd[event_scores_pd["route_id"] == route_id].sort_values("event_id")
        confidence = float(event_scores["success_probability"].prod()) if len(event_scores) else 1.0
        new_route = dict(route)
        new_route["confidence"] = confidence
        new_route["probability_events"] = event_scores.to_dict("records")
        if confidence >= min_confidence:
            scored_routes.append(new_route)
    scored_routes = sorted(
        scored_routes,
        key=lambda r: (-int(r.get("departure_seconds", 0)), -float(r.get("confidence", 0.0)), float(r.get("walk_m", 0.0))),
    )
    return scored_routes, event_scores_pd


def get_planner_for_day(conn, cfg, day: str, max_walk_m: float, from_stop: Optional[str] = None, to_stop: Optional[str] = None) -> Tuple[RobustJourneyPlanner, pd.DataFrame]:
    cache_key = (str(day).lower(), float(max_walk_m), cfg.timetable_date, tuple(cfg.region_uuids))
    if cache_key in sys._PLANNER_CACHE:
        planner, stops_pd = sys._PLANNER_CACHE[cache_key]
        # Attach in-memory tables if missing in the cached instance
        if not hasattr(planner, "_stops_pd") or not hasattr(planner, "_stop_times_pd"):
            _, _, stop_times_pd = load_planner_inputs_from_trino(
                conn,
                cfg,
                max_walk_m=max_walk_m,
                day=day,
            )
            planner._stops_pd = stops_pd
            planner._stop_times_pd = stop_times_pd
        if from_stop is not None and to_stop is not None:
            try:
                planner._resolve_stop_input(from_stop)
                planner._resolve_stop_input(to_stop)
                return planner, stops_pd
            except Exception:
                print(f"Stale cached planner detected (lacks stops: {from_stop} or {to_stop}). Rebuilding...")
                pass
        else:
            return planner, stops_pd

    stops_pd, walk_pd, stop_times_pd = load_planner_inputs_from_trino(
        conn,
        cfg,
        max_walk_m=max_walk_m,
        day=day,
    )
    planner = RobustJourneyPlanner()
    planner.prepare(
        stops_pd,
        walk_pd,
        stop_times_pd,
        max_same_trip_skip=50,
        max_walk_neighbors=12,
    )
    # Store reference to in-memory tables on the planner for zero-latency lookups
    planner._stops_pd = stops_pd
    planner._stop_times_pd = stop_times_pd
    sys._PLANNER_CACHE[cache_key] = (planner, stops_pd)
    return planner, stops_pd


def plan_robust_journey(
    *,
    spark,
    conn,
    cfg,
    reduced_model,
    residual_lookup_df: DataFrame,
    weather_df: Optional[DataFrame],
    start_stop: str,
    end_stop: str,
    arrival_time: str,
    day: str,
    service_date: str,
    requested_confidence: float = 0.80,
    max_walk_m: float = 500.0,
    max_routes: int = 50,
    max_transit_legs: int = 12,
    max_search_seconds: int = 4 * 3600,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame, RobustJourneyPlanner, pd.DataFrame]:
    """
    End-to-end robust journey planner.
    """
    planner, stops_pd = get_planner_for_day(conn, cfg, day, max_walk_m)

    candidate_routes = planner.route_reverse(
        start_stop,
        end_stop,
        arrival_time=arrival_time,
        day=day,
        max_walk_m=max_walk_m,
        max_routes=max_routes,
        max_transit_legs=max_transit_legs,
        min_transfer_seconds=cfg.min_transfer_seconds,
        max_search_seconds=max_search_seconds,
    )
    print("Candidate routes generated:", len(candidate_routes))

    if not candidate_routes:
        empty_summary = pd.DataFrame(columns=["rank", "departure_time", "arrival_deadline", "confidence", "transit_legs", "walk_m"])
        return [], empty_summary, pd.DataFrame(), planner, stops_pd

    route_events_pd = routes_to_event_rows(candidate_routes, service_date)
    if route_events_pd.empty:
        scored_routes = []
        for route in candidate_routes:
            route = dict(route)
            route["confidence"] = 1.0
            route["probability_events"] = []
            if route["confidence"] >= requested_confidence:
                scored_routes.append(route)
        route_summary_pd = pd.DataFrame([
            {
                "rank": i + 1,
                "departure_time": r["departure_time"],
                "arrival_deadline": r["arrival_deadline"],
                "confidence": r["confidence"],
                "transit_legs": r["transit_legs"],
                "walk_m": r["walk_m"],
            }
            for i, r in enumerate(scored_routes)
        ])
        return scored_routes, route_summary_pd, pd.DataFrame(), planner, stops_pd

    model_events_df = prepare_route_events_for_reduced_model(
        spark=spark,
        cfg=cfg,
        events_pd=route_events_pd,
        weather_df=weather_df,
    )
    predicted_events_df = reduced_model.transform(model_events_df)
    scored_events_df = score_events_with_residual_lookup(predicted_events_df, residual_lookup_df).cache()
    scored_routes, event_scores_pd = attach_scored_events_to_routes(candidate_routes, scored_events_df, requested_confidence)
    scored_events_df.unpersist()
    scored_routes = deduplicate_routes_for_display(scored_routes, planner)

    route_summary_pd = pd.DataFrame([
        {
            "rank": i + 1,
            "departure_time": r["departure_time"],
            "arrival_deadline": r["arrival_deadline"],
            "confidence": r["confidence"],
            "transit_legs": r["transit_legs"],
            "walk_m": r["walk_m"],
        }
        for i, r in enumerate(scored_routes)
    ])
    return scored_routes, route_summary_pd, event_scores_pd, planner, stops_pd


def confidence_color(confidence: float) -> str:
    if pd.isna(confidence):
        return "#9ca3af"
    confidence = float(confidence)
    if confidence >= 0.90:
        return "#15803d"
    if confidence >= 0.80:
        return "#65a30d"
    if confidence >= 0.65:
        return "#ca8a04"
    if confidence >= 0.50:
        return "#ea580c"
    return "#dc2626"


def resolve_stop_row(stops_pd: pd.DataFrame, stop: str) -> Optional[pd.Series]:
    if stops_pd.empty or stop is None:
        return None
    stop = str(stop)
    exact_id = stops_pd[stops_pd["stop_id"].astype(str) == stop]
    if len(exact_id):
        return exact_id.iloc[0]
    exact_name = stops_pd[stops_pd["stop_name"].astype(str).str.lower() == stop.lower()]
    if len(exact_name):
        return exact_name.iloc[0]
    partial_name = stops_pd[stops_pd["stop_name"].astype(str).str.lower().str.contains(stop.lower(), na=False)]
    if len(partial_name):
        return partial_name.iloc[0]
    return None


def fetch_intermediate_stops(conn, cfg, trip_id: str, from_id: str, to_id: str, planner: Optional[RobustJourneyPlanner] = None) -> List[Tuple[str, float, float, str]]:
    if not trip_id or trip_id == "UNKNOWN":
        return []

    rows = []
    # Try using in-memory local data first for zero latency
    if planner is not None and hasattr(planner, "_stop_times_pd") and hasattr(planner, "_stops_pd"):
        stop_times_df = planner._stop_times_pd
        stops_df = planner._stops_pd

        trip_st = stop_times_df[stop_times_df["trip_id"] == trip_id]
        if not trip_st.empty:
            stops_lookup = stops_df.drop_duplicates("stop_id")
            merged = trip_st.merge(stops_lookup, on="stop_id", how="inner")
            merged = merged.sort_values("stop_sequence")
            for _, row in merged.iterrows():
                rows.append((
                    str(row["stop_id"]),
                    str(row["stop_name"]),
                    float(row["stop_lat"]),
                    float(row["stop_lon"]),
                    int(row["stop_sequence"])
                ))

    # Fallback to Trino query if local data is not found or empty
    if not rows:
        sharedNS = "iceberg.com490_iceberg"
        sql = f"""
        SELECT DISTINCT st.stop_id, s.stop_name, s.stop_lat, s.stop_lon, st.stop_sequence
        FROM {sharedNS}.sbb_stop_times st
        JOIN {sharedNS}.sbb_stops s
          ON st.stop_id = s.stop_id AND st.pub_date = s.pub_date
        WHERE st.trip_id = '{trip_id}'
          AND st.pub_date = DATE '{cfg.timetable_date}'
        ORDER BY st.stop_sequence
        """
        try:
            with closing(conn.cursor()) as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
        except Exception as e:
            print("Trino intermediate stops query failed:", str(e))
            return []

    if not rows:
        return []

    # Map sequences to BPUIC to find range
    from_bpuic = str(from_id).split(":")[0]
    to_bpuic = str(to_id).split(":")[0]

    from_seq = None
    to_seq = None

    for r in rows:
        r_bpuic = str(r[0]).split(":")[0]
        if r_bpuic == from_bpuic:
            from_seq = int(r[4])
        if r_bpuic == to_bpuic:
            to_seq = int(r[4])

    if from_seq is None or to_seq is None:
        for r in rows:
            if str(r[0]).startswith(from_bpuic) and from_seq is None:
                from_seq = int(r[4])
            if str(r[0]).startswith(to_bpuic) and to_seq is None:
                to_seq = int(r[4])

    if from_seq is None or to_seq is None:
        return [(str(r[0]), float(r[2]), float(r[3]), str(r[1])) for r in rows]

    start_seq, end_seq = min(from_seq, to_seq), max(from_seq, to_seq)

    filtered_stops = []
    for r in rows:
        seq = int(r[4])
        if start_seq <= seq <= end_seq:
            filtered_stops.append((str(r[0]), float(r[2]), float(r[3]), str(r[1])))

    if from_seq > to_seq:
        filtered_stops = filtered_stops[::-1]

    return filtered_stops


def plot_route_itinerary_map(route: Mapping[str, Any], stops_pd: pd.DataFrame, planner: RobustJourneyPlanner, conn, cfg) -> str:
    import json
    import html
    import base64

    # If route is empty, return a clean blank map
    if not route or "steps" not in route:
        blank_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
            <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
            <style>
                html, body, #map { height: 100%; margin: 0; padding: 0; background: #f8f9fa; }
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map', {
                    zoomControl: true,
                    zoomAnimation: false,
                    fadeAnimation: false,
                    inertia: false
                }).setView([46.5197, 6.6323], 13);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    maxZoom: 17, minZoom: 11,
                    attribution: '&copy; OpenStreetMap'
                }).addTo(map);
            </script>
        </body>
        </html>
        """
        b64_html = base64.b64encode(blank_html.encode('utf-8')).decode('utf-8')
        return f'<iframe src="data:text/html;charset=utf-8;base64,{b64_html}" width="100%" height="380px" style="border:none !important; overflow:hidden;" scrolling="no"></iframe>'

    # Build unique stop coordinate lookup dictionary
    stops = stops_pd.drop_duplicates("stop_id").copy()
    stops["stop_id"] = stops["stop_id"].astype(str)
    stop_lookup = stops.set_index("stop_id")

    coords = []
    steps = route.get("steps", [])
    if not steps:
        blank_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
            <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
            <style>
                html, body, #map { height: 100%; margin: 0; padding: 0; background: #f8f9fa; }
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map', {
                    zoomControl: true, zoomAnimation: false, fadeAnimation: false, inertia: false
                }).setView([46.5197, 6.6323], 13);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    maxZoom: 17, minZoom: 11,
                    attribution: '&copy; OpenStreetMap'
                }).addTo(map);
            </script>
        </body>
        </html>
        """
        b64_html = base64.b64encode(blank_html.encode('utf-8')).decode('utf-8')
        return f'<iframe src="data:text/html;charset=utf-8;base64,{b64_html}" width="100%" height="380px" style="border:none !important; overflow:hidden;" scrolling="no"></iframe>'

    start_stop_id = str(steps[0]["from"])
    end_stop_id = str(steps[-1]["to"])
    map_legs = []

    for idx, step in enumerate(steps):
        from_id = str(step["from"])
        to_id = str(step["to"])

        if step.get("type") == "TRANSIT":
            trip_id = step.get("trip_id")
            stops_list = fetch_intermediate_stops(conn, cfg, trip_id, from_id, to_id, planner)
            if stops_list:
                map_legs.append({
                    "type": "TRANSIT",
                    "stops": stops_list,
                    "line_text": str(step.get("line_text", "Transit")).upper(),
                    "dep_time": step.get("dep_time_str"),
                    "arr_time": step.get("arr_time_str")
                })
                for s in stops_list:
                    coords.append((float(s[1]), float(s[2])))
            else:
                if from_id in stop_lookup.index and to_id in stop_lookup.index:
                    a = stop_lookup.loc[from_id]
                    b = stop_lookup.loc[to_id]
                    fallback_stops = [
                        (from_id, float(a["stop_lat"]), float(a["stop_lon"]), planner._name(from_id)),
                        (to_id, float(b["stop_lat"]), float(b["stop_lon"]), planner._name(to_id))
                    ]
                    map_legs.append({
                        "type": "TRANSIT",
                        "stops": fallback_stops,
                        "line_text": str(step.get("line_text", "Transit")).upper(),
                        "dep_time": step.get("dep_time_str"),
                        "arr_time": step.get("arr_time_str")
                    })
                    coords.append((float(a["stop_lat"]), float(a["stop_lon"])))
                    coords.append((float(b["stop_lat"]), float(b["stop_lon"])))
        else: # WALK
            if from_id in stop_lookup.index and to_id in stop_lookup.index:
                a = stop_lookup.loc[from_id]
                b = stop_lookup.loc[to_id]
                walk_stops = [
                    (from_id, float(a["stop_lat"]), float(a["stop_lon"]), planner._name(from_id)),
                    (to_id, float(b["stop_lat"]), float(b["stop_lon"]), planner._name(to_id))
                ]
                map_legs.append({
                    "type": "WALK",
                    "stops": walk_stops,
                    "distance_m": step.get("distance_m", 0)
                })
                coords.append((float(a["stop_lat"]), float(a["stop_lon"])))
                coords.append((float(b["stop_lat"]), float(b["stop_lon"])))

    # Calculate center
    if coords:
        center_lat = sum(c[0] for c in coords) / len(coords)
        center_lon = sum(c[1] for c in coords) / len(coords)
    else:
        center_lat, center_lon = 46.5197, 6.6323

    # Construct clean arrays to serialize to JS
    js_legs = []
    drawn_stops = set()
    js_markers = []

    for leg in map_legs:
        leg_stops = leg["stops"]
        if not leg_stops:
            continue

        points = [[float(s[1]), float(s[2])] for s in leg_stops]

        if leg["type"] == "TRANSIT":
            popup_text = f"Line: {leg['line_text']} ({leg['dep_time']} - {leg['arr_time']})"
            js_legs.append({
                "type": "TRANSIT",
                "points": points,
                "color": "#1a73e8",
                "weight": 5.5,
                "opacity": 0.95,
                "popup": popup_text
            })
        else:
            popup_text = f"Walk {leg['distance_m']:.0f}m"
            js_legs.append({
                "type": "WALK",
                "points": points,
                "color": "#70757a",
                "weight": 4.5,
                "opacity": 0.9,
                "popup": popup_text
            })

        for idx, (stop_id, lat, lon, name) in enumerate(leg_stops):
            stop_id = str(stop_id)
            if stop_id in drawn_stops:
                continue

            # Start marker
            if stop_id == start_stop_id:
                js_markers.append({
                    "type": "START",
                    "lat": float(lat),
                    "lon": float(lon),
                    "name": name,
                    "popup": f"<b>Start: {name}</b>"
                })
                drawn_stops.add(stop_id)
            # End marker
            elif stop_id == end_stop_id:
                js_markers.append({
                    "type": "END",
                    "lat": float(lat),
                    "lon": float(lon),
                    "name": name,
                    "popup": f"<b>Destination: {name}</b>"
                })
                drawn_stops.add(stop_id)
            # Intermediate marker
            else:
                details = [f"<b>{name}</b>"]
                is_transfer = False
                if leg["type"] == "TRANSIT":
                    if idx == 0:
                        details.append(f"Boarding Stop | Dep: {leg['dep_time']}")
                        is_transfer = True
                    elif idx == len(leg_stops) - 1:
                        details.append(f"Alighting Stop | Arr: {leg['arr_time']}")
                        is_transfer = True
                    else:
                        details.append("Intermediate stop")

                js_markers.append({
                    "type": "TRANSFER" if is_transfer else "INTERMEDIATE",
                    "lat": float(lat),
                    "lon": float(lon),
                    "name": name,
                    "popup": "<br>".join(details)
                })
                drawn_stops.add(stop_id)

    # Generate pure, high-performance HTML avoiding heavy jquery/bootstrap packages
    leaflet_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
        <style>
            html, body, #map {{
                height: 100%;
                width: 100%;
                margin: 0;
                padding: 0;
                background: #f8f9fa;
            }}
            .leaflet-popup-content {{
                font-family: sans-serif;
                font-size: 11px;
                color: #2c3e50;
                line-height: 1.4;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = L.map('map', {{
                preferCanvas: true,
                zoomControl: true,
                zoomAnimation: false,
                fadeAnimation: false,
                inertia: false
            }}).setView([{center_lat}, {center_lon}], 13);

            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 17,
                minZoom: 11,
                attribution: '&copy; OpenStreetMap'
            }}).addTo(map);

            var legsData = {json.dumps(js_legs)};
            var markersData = {json.dumps(js_markers)};
            var allLatLngs = [];

            // 1. Draw Legs
            legsData.forEach(function(leg) {{
                var pts = leg.points;
                pts.forEach(function(p) {{
                    allLatLngs.push(p);
                }});

                // Shadow polyline for 3D depth elevation
                var shadowWeight = leg.type === 'TRANSIT' ? 10 : 8;
                L.polyline(pts, {{
                    color: '#0f172a',
                    weight: shadowWeight,
                    opacity: 0.12
                }}).addTo(map);

                // Main polyline
                var options = {{
                    color: leg.color,
                    weight: leg.weight,
                    opacity: leg.opacity
                }};
                if (leg.type === 'WALK') {{
                    options.dashArray = '4, 8';
                }}
                var poly = L.polyline(pts, options).addTo(map);
                poly.bindPopup(leg.popup);
            }});

            // 2. Draw Stop Markers
            markersData.forEach(function(marker) {{
                var latlng = [marker.lat, marker.lon];

                if (marker.type === 'START') {{
                    var startSvg = `
                    <svg width="32" height="40" viewBox="0 0 32 40" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0px 3px 5px rgba(0,0,0,0.35));">
                      <path d="M16 0C7.16 0 0 7.16 0 16C0 26.4 16 40 16 40C16 40 32 26.4 32 16C32 7.16 24.84 0 16 0Z" fill="#34a853"/>
                      <circle cx="16" cy="16" r="6" fill="white"/>
                      <circle cx="16" cy="16" r="3" fill="#34a853"/>
                    </svg>`;
                    L.marker(latlng, {{
                        icon: L.divIcon({{
                            className: '',
                            iconSize: [32, 40],
                            iconAnchor: [16, 40],
                            html: startSvg
                        }}),
                        tooltip: "Start: " + marker.name
                    }}).addTo(map).bindPopup(marker.popup);

                }} else if (marker.type === 'END') {{
                    var endSvg = `
                    <svg width="32" height="40" viewBox="0 0 32 40" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0px 3px 5px rgba(0,0,0,0.35));">
                      <path d="M16 0C7.16 0 0 7.16 0 16C0 26.4 16 40 16 40C16 40 32 26.4 32 16C32 7.16 24.84 0 16 0Z" fill="#ea4335"/>
                      <circle cx="16" cy="16" r="6" fill="white"/>
                      <circle cx="16" cy="16" r="3" fill="#ea4335"/>
                    </svg>`;
                    L.marker(latlng, {{
                        icon: L.divIcon({{
                            className: '',
                            iconSize: [32, 40],
                            iconAnchor: [16, 40],
                            html: endSvg
                        }}),
                        tooltip: "Destination: " + marker.name
                    }}).addTo(map).bindPopup(marker.popup);

                }} else if (marker.type === 'TRANSFER') {{
                    L.circleMarker(latlng, {{
                        radius: 8,
                        color: '#ea580c',
                        weight: 3.5,
                        fillColor: '#ffffff',
                        fillOpacity: 1.0,
                        tooltip: "Transfer: " + marker.name
                    }}).addTo(map).bindPopup(marker.popup);

                }} else {{
                    L.circleMarker(latlng, {{
                        radius: 4.5,
                        color: '#1a73e8',
                        weight: 2.5,
                        fillColor: '#ffffff',
                        fillOpacity: 1.0,
                        tooltip: marker.name
                    }}).addTo(map).bindPopup(marker.popup);
                }}
            }});

            if (allLatLngs.length > 0) {{
                map.fitBounds(allLatLngs);
            }}
        </script>
    </body>
    </html>
    """

    b64_html = base64.b64encode(leaflet_html.encode('utf-8')).decode('utf-8')
    iframe_html = f'<iframe src="data:text/html;charset=utf-8;base64,{b64_html}" width="100%" height="380px" style="border:none !important; overflow:hidden;" scrolling="no"></iframe>'
    return iframe_html


# --- Interactive Jupyter widgets dashboard class ---

class RobustJourneyDashboard:
    """
    High-fidelity ipywidgets dashboard for SBB Robust Journey Planning.
    """
    def __init__(
        self,
        spark,
        conn,
        cfg,
        reduced_model,
        residual_lookup_df,
        weather_df=None,
    ):
        self.spark = spark
        self.conn = conn
        self.cfg = cfg
        self.reduced_model = reduced_model
        self.residual_lookup_df = residual_lookup_df
        self.weather_df = weather_df

        # Distribute the lightweight UDF file to Spark workers so they can deserialize our UDFs
        try:
            self.spark.sparkContext.addPyFile("/home/bdoan/final/robust_udfs.py")
            print("Successfully distributed robust_udfs.py to Spark executors.")
        except Exception as e:
            print("Warning: Could not distribute robust_udfs.py to Spark executors:", str(e))

        self.routes = []
        self.summary = pd.DataFrame()
        self.planner = None
        self.stops_pd = pd.DataFrame()
        self._updating_widgets = False
        self.residual_lookup_pd = residual_lookup_df.toPandas()

        # Fetch all stops that are served by trips touching the Lausanne region
        cache_key = (cfg.timetable_date, cfg.region_name)
        if cache_key in sys._ALL_STOP_NAMES_CACHE:
            self.all_stop_names = sys._ALL_STOP_NAMES_CACHE[cache_key]
            print(f"Loaded {len(self.all_stop_names)} stop names from sys cache.")
        else:
            print("Pre-fetching network stops (including Renens, Ecublens, Epalinges)...")
            sharedNS = "iceberg.com490_iceberg"
            region_table = f"iceberg.{cfg.username}.region_stops_{cfg.region_name}"
            stops_sql = f"""
            WITH route_stops AS (
                SELECT DISTINCT st.stop_id
                FROM {sharedNS}.sbb_stop_times st
                JOIN (
                    SELECT DISTINCT st2.trip_id
                    FROM {sharedNS}.sbb_stop_times st2
                    JOIN {region_table} rs
                      ON st2.stop_id = rs.stop_id
                    WHERE st2.pub_date = DATE '{cfg.timetable_date}'
                ) tr ON st.trip_id = tr.trip_id
                WHERE st.pub_date = DATE '{cfg.timetable_date}'
            )
            SELECT DISTINCT s.stop_name
            FROM {sharedNS}.sbb_stops s
            JOIN route_stops rs
              ON s.stop_id = rs.stop_id
            WHERE s.pub_date = DATE '{cfg.timetable_date}'
              AND NOT s.stop_id LIKE 'Parent%'
            ORDER BY s.stop_name
            """
            with closing(conn.cursor()) as cursor:
                cursor.execute(stops_sql)
                all_stops_rows = cursor.fetchall()
            self.all_stop_names = [row[0] for row in all_stops_rows]
            sys._ALL_STOP_NAMES_CACHE[cache_key] = self.all_stop_names
            print(f"Loaded {len(self.all_stop_names)} stop names for autocomplete.")

        # UI components
        import ipywidgets as widgets
        import datetime

        # Styles - ultra compact title
        self.title_html = widgets.HTML(
            value='<div style="display: flex; align-items: baseline; margin-bottom: 2px;">'
                  '  <h1 style="color: #2c3e50; font-family: \'Outfit\', sans-serif; font-size: 20px; margin: 0; padding-right: 12px;">🇨🇭 SBB Robust Journey Planner</h1>'
                  '  <span style="color: #7f8c8d; font-family: sans-serif; font-size: 11px;">Predictive delay modeling & stochastic route planner</span>'
                  '</div>'
        )

        # Main control forms
        self.area_dd = widgets.Dropdown(
            options=["Lausanne"],
            value="Lausanne",
            description="Area:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='95%')
        )

        self.date_picker = widgets.DatePicker(
            value=datetime.date(2026, 1, 25),
            description="Travel Date:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='95%')
        )

        # Fallback if list is empty
        if not self.all_stop_names:
            self.all_stop_names = ["Renens VD, gare", "Epalinges, Croisettes"]

        # Resolve From stop dynamically to EPFL
        default_from = "Ecublens VD, EPFL"
        for name in self.all_stop_names:
            lower_name = name.lower()
            if "epfl" in lower_name:
                default_from = name
                break
        if default_from not in self.all_stop_names:
            default_from = self.all_stop_names[0]

        # Resolve To stop dynamically to Ouchy
        default_to = "Lausanne, Ouchy-Olympique"
        for name in self.all_stop_names:
            lower_name = name.lower()
            if "ouchy-olympique" in lower_name:
                default_to = name
                break
        if default_to not in self.all_stop_names:
            for name in self.all_stop_names:
                lower_name = name.lower()
                if "ouchy" in lower_name:
                    default_to = name
                    break
        if default_to not in self.all_stop_names:
            default_to = self.all_stop_names[min(1, len(self.all_stop_names) - 1)]

        # Dropdowns preloaded with all stops and defaults to resolved Renens / Epalinges
        self.from_box = widgets.Dropdown(
            options=self.all_stop_names,
            value=default_from,
            description="From Stop:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='95%')
        )
        self.to_box = widgets.Dropdown(
            options=self.all_stop_names,
            value=default_to,
            description="To Stop:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='95%')
        )
        self.arr_time = widgets.Text(
            value="09:00",
            description="Arrive by:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='95%')
        )
        self.conf_slider = widgets.FloatSlider(
            value=0.80,
            min=0.00,
            max=1.00,
            step=0.05,
            description="Confidence:",
            orientation='horizontal',
            style={'description_width': 'initial'},
            readout=True,
            readout_format='.2f',
            tooltip="Confidence level (Q%): The required probability of arriving at your destination on time.",
            layout=widgets.Layout(width='95%')
        )
        self.walk_slider = widgets.IntSlider(
            value=500,
            min=100,
            max=2000,
            step=100,
            description="Max Walk (m):",
            orientation='horizontal',
            style={'description_width': 'initial'},
            readout=True,
            readout_format='d',
            tooltip="Max Walk: The maximum total walking distance allowed for transfers in meters.",
            layout=widgets.Layout(width='95%')
        )

        self.btn_search = widgets.Button(
            description="Find Routes",
            button_style='primary',
            icon='search',
            layout=widgets.Layout(width='95%', height='32px', margin='0px')
        )
        self.btn_search.on_click(self.on_search_clicked)

        # Progress bar for step-by-step activity
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=10,
            step=1,
            description='Loading:',
            bar_style='info',
            orientation='horizontal',
            layout=widgets.Layout(width='100%', height='18px', visibility='hidden')
        )

        # Output/display areas - reduced height for video demos
        self.route_select = widgets.Select(
            options=["No search conducted yet. Click 'Find Routes' to begin."],
            description="",
            layout=widgets.Layout(width='100%', height='80px')
        )
        self.route_select.observe(self.on_route_selected, names='value')

        self.details_html = widgets.HTML(
            value='<div style="padding: 10px; background-color: #f8f9fa; border-radius: 6px; font-family: sans-serif; font-size: 12px; color: #7f8c8d;">Search for a route above to see details...</div>',
            layout=widgets.Layout(width='100%', height='255px', overflow='auto')
        )

        self.map_html = widgets.HTML(
            value='<div style="padding: 10px; background-color: #f8f9fa; border-radius: 6px; font-family: sans-serif; color: #7f8c8d; text-align: center; margin-top: 100px;">Search for a route above to render interactive map...</div>',
            layout=widgets.Layout(width='100%', height='380px', overflow='hidden')
        )
        self.status_html = widgets.HTML(value='', layout=widgets.Layout(margin='0px'))

    def on_search_clicked(self, b):
        self.btn_search.disabled = True
        self.progress_bar.layout.visibility = 'visible'
        self.progress_bar.value = 1
        self.progress_bar.description = "Timetable:"
        self.status_html.value = '<div style="color: #2980b9; font-weight: bold; font-size: 11px; padding: 2px 0;">⏳ Step 1/3: Loading SBB timetable and building routing graph (takes ~30-40s on first run)...</div>'
        try:
            # 1. Dynamically infer day of week from DatePicker object
            try:
                if self.date_picker.value is None:
                    raise ValueError("No date selected")
                date_val = pd.to_datetime(self.date_picker.value)
                inferred_day = date_val.day_name().lower()
                travel_date_str = date_val.strftime('%Y-%m-%d')
            except Exception:
                self.status_html.value = '<div style="color: #c0392b; font-weight: bold; font-size: 11px; padding: 2px 0;">⚠️ Please select a valid travel date from the calendar.</div>'
                self.btn_search.disabled = False
                self.progress_bar.layout.visibility = 'hidden'
                return

            # Check memory cache first to make the demo run in 0.001 seconds!
            import sys
            if not hasattr(sys, "_DASHBOARD_ROUTING_CACHE"):
                sys._DASHBOARD_ROUTING_CACHE = {}

            cache_key = (
                str(self.from_box.value),
                str(self.to_box.value),
                str(travel_date_str),
                str(self.arr_time.value),
                float(self.conf_slider.value),
                int(self.walk_slider.value)
            )

            if cache_key in sys._DASHBOARD_ROUTING_CACHE:
                print("LOADED precomputed search results from memory cache instantly!")
                cached_routes, cached_planner, cached_stops_pd = sys._DASHBOARD_ROUTING_CACHE[cache_key]
                self.routes = cached_routes
                self.planner = cached_planner
                self.stops_pd = cached_stops_pd

                # Render options
                if not self.routes:
                    self.route_select.options = [("No routes meet this confidence criteria", -1)]
                    self.details_html.value = '<div style="padding: 10px; color: #c0392b; font-size: 12px;">No journeys matching this path are guaranteed to reach the destination with the desired confidence probability. Try decreasing the confidence bar or increasing walk distance.</div>'
                    self.status_html.value = f'<div style="color: #c0392b; font-weight: bold; font-size: 11px; padding: 2px 0;">❌ No robust routes found matching confidence threshold. 💡 Try sliding "Confidence" down or increasing "Max Walk".</div>'
                    self.map_html.value = '<div style="padding: 10px; background-color: #f8f9fa; border-radius: 6px; font-family: sans-serif; color: #7f8c8d; text-align: center; margin-top: 100px;">No maps to display...</div>'
                else:
                    options = []
                    for i, r in enumerate(self.routes):
                        label = f"Dep {r['departure_time']} ➔ Arr {r['steps'][-1]['arr_time_str']} | Conf: {r['confidence'] * 100:.0f}% | Trans: {r['transit_legs'] - 1} | Walk: {r['walk_m']:.0f}m"
                        options.append((label, i))
                    self._updating_widgets = True
                    self.route_select.options = options
                    self.route_select.value = 0
                    self._updating_widgets = False

                    import time
                    self._last_route_select_time = time.time()
                    self._last_rendered_idx = 0

                    self.status_html.value = f'<div style="color: #27ae60; font-weight: bold; font-size: 11px; padding: 2px 0;">✅ Found {len(self.routes)} routes instantly from cache!</div>'
                    self.update_route_details(0)

                self.progress_bar.value = 10
                self.progress_bar.description = "Completed!"
                self.btn_search.disabled = False
                # Briefly sleep to let user see progress complete, then hide it
                time.sleep(0.1)
                self.progress_bar.layout.visibility = 'hidden'
                return

            # 2. Build or load schedule planner (graph assembly)
            planner, stops_pd = get_planner_for_day(
                self.conn,
                self.cfg,
                inferred_day,
                self.walk_slider.value,
                from_stop=self.from_box.value,
                to_stop=self.to_box.value
            )
            self.planner = planner
            self.stops_pd = stops_pd

            # Step 2: Routing Candidates Scan
            self.progress_bar.value = 4
            self.progress_bar.description = "Routing:"
            self.status_html.value = '<div style="color: #2980b9; font-weight: bold; font-size: 11px; padding: 2px 0;">⏳ Step 2/3: Executing CSA graph routing candidate search...</div>'

            candidate_routes = planner.route_reverse(
                self.from_box.value,
                self.to_box.value,
                arrival_time=self.arr_time.value,
                day=inferred_day,
                max_walk_m=self.walk_slider.value,
                max_routes=50,
                max_transit_legs=12,
                min_transfer_seconds=self.cfg.min_transfer_seconds,
                max_search_seconds=4 * 3600,
            )

            # Step 3: Spark ML scoring
            self.progress_bar.value = 7
            self.progress_bar.description = "Scoring:"
            self.status_html.value = '<div style="color: #2980b9; font-weight: bold; font-size: 11px; padding: 2px 0;">⏳ Step 3/3: Running GBT machine learning delay predictions & risk evaluations...</div>'

            self.stops_pd = stops_pd

            if not candidate_routes:
                self.routes = []
                self.route_select.options = [("No routes meet this confidence criteria", -1)]
                self.details_html.value = '<div style="padding: 10px; color: #c0392b; font-size: 12px;">No journeys matching this path are guaranteed to reach the destination with the desired confidence probability. Try decreasing the confidence bar or increasing walk distance.</div>'
                self.status_html.value = '<div style="color: #c0392b; font-weight: bold; font-size: 11px; padding: 2px 0;">❌ No timetable-feasible candidate routes found (0 paths). 💡 Double check stop names, date, or time.</div>'
                self.map_html.value = '<div style="padding: 10px; background-color: #f8f9fa; border-radius: 6px; font-family: sans-serif; color: #7f8c8d; text-align: center; margin-top: 100px;">No maps to display...</div>'
            else:
                route_events_pd = routes_to_event_rows(candidate_routes, travel_date_str)
                if route_events_pd.empty:
                    # Pure walking
                    self.routes = []
                    for r in candidate_routes:
                        r = dict(r)
                        r["confidence"] = 1.0
                        r["probability_events"] = []
                        if r["confidence"] >= self.conf_slider.value:
                            self.routes.append(r)
                else:
                    model_events_df = prepare_route_events_for_reduced_model(
                        spark=self.spark,
                        cfg=self.cfg,
                        events_pd=route_events_pd,
                        weather_df=self.weather_df,
                    )
                    predicted_events_df = self.reduced_model.transform(model_events_df)
                    predicted_events_pd = predicted_events_df.toPandas()
                    scored_events_pd = score_events_with_residual_lookup_pandas(predicted_events_pd, self.residual_lookup_pd)
                    self.routes, event_scores_pd = attach_scored_events_to_routes(candidate_routes, scored_events_pd, self.conf_slider.value)

                self.routes = deduplicate_routes_for_display(self.routes, self.planner)

                if not self.routes:
                    self.route_select.options = [("No routes meet this confidence criteria", -1)]
                    self.details_html.value = '<div style="padding: 10px; color: #c0392b; font-size: 12px;">No journeys matching this path are guaranteed to reach the destination with the desired confidence probability. Try decreasing the confidence bar or increasing walk distance.</div>'
                    self.status_html.value = f'<div style="color: #c0392b; font-weight: bold; font-size: 11px; padding: 2px 0;">❌ No robust routes found matching confidence threshold (out of {len(candidate_routes)} timetable paths). 💡 Try sliding "Confidence" down to 0.50 or increasing "Max Walk".</div>'
                    self.map_html.value = '<div style="padding: 10px; background-color: #f8f9fa; border-radius: 6px; font-family: sans-serif; color: #7f8c8d; text-align: center; margin-top: 100px;">No maps to display...</div>'
                else:
                    options = []
                    for i, r in enumerate(self.routes):
                        label = f"Dep {r['departure_time']} ➔ Arr {r['steps'][-1]['arr_time_str']} | Conf: {r['confidence'] * 100:.0f}% | Trans: {r['transit_legs'] - 1} | Walk: {r['walk_m']:.0f}m"
                        options.append((label, i))
                    self._updating_widgets = True
                    self.route_select.options = options
                    self.route_select.value = 0
                    self._updating_widgets = False

                    # Update debouncer state to reflect the direct render call
                    import time
                    self._last_route_select_time = time.time()
                    self._last_rendered_idx = 0

                    self.status_html.value = f'<div style="color: #27ae60; font-weight: bold; font-size: 11px; padding: 2px 0;">✅ Found {len(self.routes)} routes (Date: {travel_date_str})!</div>'
                    self.update_route_details(0)

            # Store successful search run in the memory cache
            if 'travel_date_str' in locals() and hasattr(self, "routes") and hasattr(self, "planner") and hasattr(self, "stops_pd"):
                cache_key = (
                    str(self.from_box.value),
                    str(self.to_box.value),
                    str(travel_date_str),
                    str(self.arr_time.value),
                    float(self.conf_slider.value),
                    int(self.walk_slider.value)
                )
                import sys
                if not hasattr(sys, "_DASHBOARD_ROUTING_CACHE"):
                    sys._DASHBOARD_ROUTING_CACHE = {}
                sys._DASHBOARD_ROUTING_CACHE[cache_key] = (self.routes, self.planner, self.stops_pd)
                print("SAVED search results to memory cache!")

            self.progress_bar.value = 10
            self.progress_bar.description = "Completed!"
        except Exception as e:
            self.status_html.value = f'<div style="color: #c0392b; font-weight: bold; font-size: 11px; padding: 2px 0;">⚠️ Error: {str(e)}</div>'
            import traceback
            traceback.print_exc()
        finally:
            self.btn_search.disabled = False
            # Briefly sleep to let user see progress complete, then hide it
            import time
            self.progress_bar.layout.visibility = 'hidden'

    def on_route_selected(self, change):
        idx = change['new']
        if idx is None or idx < 0:
            return

        # Robust timestamp and index-based event debouncer
        import time
        now = time.time()
        if hasattr(self, "_last_route_select_time") and (now - self._last_route_select_time) < 0.8:
            if hasattr(self, "_last_rendered_idx") and self._last_rendered_idx == idx:
                return

        self._last_route_select_time = now
        self._last_rendered_idx = idx
        self.update_route_details(idx)

    def update_route_details(self, idx):
        if idx < 0 or idx >= len(self.routes):
            return
        route = self.routes[idx]

        # Build timeline HTML representation - extremely compact
        html = '<div style="font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif; background-color: #ffffff; padding: 10px; border-radius: 6px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">'

        # Header block - perfectly aligned Title on the left, Confidence Badge on the right
        color = confidence_color(route['confidence'])
        html += f'<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 4px; margin-bottom: 8px;">'
        html += f'  <span style="font-size: 12px; font-weight: bold; color: #2c3e50; font-family: sans-serif;">Journey Timeline</span>'
        html += f'  <div style="background-color: {color}; color: white; padding: 2px 7px; border-radius: 10px; font-weight: bold; font-size: 10px;">'
        html += f'    Confidence: {route["confidence"] * 100:.0f}%'
        html += f'  </div>'
        html += f'</div>'

        # Steps list
        html += '<div style="position: relative; padding-left: 16px; border-left: 2px solid #cbd5e1; margin-left: 8px;">'

        for s_idx, step in enumerate(route['steps']):
            # Dot on the vertical line
            step_color = "#2563eb" if step["type"] == "TRANSIT" else "#64748b"
            html += f'<div style="position: absolute; left: -6px; width: 10px; height: 10px; border-radius: 50%; background-color: {step_color}; border: 2px solid #ffffff; box-shadow: 0 0 0 1.5px {step_color};"></div>'

            # Step block
            html += f'<div style="margin-bottom: 12px; padding-left: 10px;">'

            if step["type"] == "TRANSIT":
                line_label = str(step.get("line_text", "UNKNOWN")).upper()
                operator_label = f'<span style="background-color: #fee2e2; color: #dc2626; padding: 1px 4px; border-radius: 3px; font-size: 10px; font-weight: bold; margin-right: 4px;">{line_label}</span>'
                html += f'  <div style="display: flex; justify-content: space-between; align-items: center;">'
                html += f'    <span style="font-weight: bold; font-size: 12px; color: #0f172a;">{operator_label} {self.planner._name(step["from"])} ➔ {self.planner._name(step["to"])}</span>'
                html += f'    <span style="font-size: 11px; color: #64748b;">({step["dep_time_str"]} - {step["arr_time_str"]})</span>'
                html += f'  </div>'

                # If there's scored delay model events corresponding to this transfer
                matching_event = None
                for ev in route.get("probability_events", []):
                    if str(ev.get("from_bpuic")) == str(step["from"]).split(":")[0] and str(ev.get("to_bpuic")) == str(step["to"]).split(":")[0]:
                        matching_event = ev
                        break

                if matching_event:
                    prediction = matching_event.get("prediction", 0.0)
                    slack = matching_event.get("usable_slack_min", 0.0)
                    prob = matching_event.get("success_probability", 1.0)
                    level = matching_event.get("residual_level_used", "global")
                    html += f'  <div style="font-size: 10.5px; color: #475569; margin-top: 2px; background-color: #f8fafc; padding: 4px 6px; border-radius: 4px; border: 1px solid #f1f5f9;">'
                    html += f'    💡 <b>Segment Risk:</b> GBT Delay: <b>+{prediction:.1f}m</b> | Usable slack: <b>{slack:.1f}m</b> | Success Prob: <b style="color: {confidence_color(prob)}">{prob * 100:.0f}%</b> (CDF: <i>{level}</i>)'
                    html += f'  </div>'
            else: # WALK
                html += f'  <div style="display: flex; justify-content: space-between; align-items: center;">'
                html += f'    <span style="font-weight: 500; font-size: 11.5px; color: #475569;">🚶 Walk {step["distance_m"]:.0f} meters: {self.planner._name(step["from"])} ➔ {self.planner._name(step["to"])}</span>'
                html += f'    <span style="font-size: 11px; color: #64748b;">(~{int(step["distance_m"]/0.83)}s)</span>'
                html += f'  </div>'

            html += f'</div>'

        # Final arrival node to complete the journey timeline visually!
        last_step = route['steps'][-1]
        dest_name = self.planner._name(last_step['to'])
        arrival_time = last_step['arr_time_str']
        deadline_time = route['arrival_deadline']

        # Dot on the vertical line (destination red/checkered style)
        html += f'<div style="position: absolute; left: -6px; width: 10px; height: 10px; border-radius: 50%; background-color: #ef4444; border: 2px solid #ffffff; box-shadow: 0 0 0 1.5px #ef4444;"></div>'
        html += f'<div style="margin-bottom: 2px; padding-left: 10px;">'
        html += f'  <div style="display: flex; justify-content: space-between; align-items: center;">'
        html += f'    <span style="font-weight: bold; font-size: 12px; color: #0f172a;">🏁 Arrive at {dest_name}</span>'
        html += f'    <span style="font-size: 11px; color: #64748b; font-weight: bold;">(Arr: {arrival_time} | Deadline: {deadline_time})</span>'
        html += f'  </div>'
        html += f'</div>'

        html += '</div>' # End of timeline list
        html += '</div>' # End of main card

        self.details_html.value = html

        # Update Map
        try:
            iframe_html = plot_route_itinerary_map(
                route=route,
                stops_pd=self.stops_pd,
                planner=self.planner,
                conn=self.conn,
                cfg=self.cfg
            )
            self.map_html.value = iframe_html
        except Exception as e:
            self.map_html.value = f'<div style="color: #dc2626; padding: 10px; font-size: 12px;">Failed to render interactive map: {str(e)}</div>'

    def show(self):
        import ipywidgets as widgets

        # Set precise visual widths for each widget inside horizontal flex rows
        self.from_box.layout = widgets.Layout(width='28%')
        self.to_box.layout = widgets.Layout(width='28%')
        self.date_picker.layout = widgets.Layout(width='24%')
        self.arr_time.layout = widgets.Layout(width='16%')

        self.conf_slider.layout = widgets.Layout(width='33%')
        self.walk_slider.layout = widgets.Layout(width='33%')
        self.area_dd.layout = widgets.Layout(width='16%')
        self.btn_search.layout = widgets.Layout(width='14%', height='28px', margin='0px') # perfect inline alignment

        row1 = widgets.HBox(
            [self.from_box, self.to_box, self.date_picker, self.arr_time],
            layout=widgets.Layout(width='100%', justify_content='space-between')
        )
        row2 = widgets.HBox(
            [self.conf_slider, self.walk_slider, self.area_dd, self.btn_search],
            layout=widgets.Layout(width='100%', justify_content='space-between', align_items='center')
        )

        # Wrap in a neat controls VBox
        controls_box = widgets.VBox([row1, row2], layout=widgets.Layout(width='100%', gap='2px'))

        # Wrap available routes list AND its header in a beautiful unified card matching details_html card style
        routes_title_html = widgets.HTML(
            value='<div style="border-bottom: 2px solid #e2e8f0; padding-bottom: 4px; margin-bottom: 8px;">'
                  '  <span style="font-size: 12px; font-weight: bold; color: #2c3e50; font-family: sans-serif;">Available Routes</span>'
                  '</div>'
        )
        self.route_select.layout = widgets.Layout(width='100%', height='72px', border='none')
        routes_card = widgets.VBox(
            [routes_title_html, self.route_select],
            layout=widgets.Layout(
                width='100%',
                padding='10px 10px 5px 10px',
                background_color='#ffffff',
                border='1px solid #e2e8f0',
                border_radius='6px',
                box_shadow='0 2px 4px rgba(0,0,0,0.05)',
                margin='0 0 10px 0'
            )
        )

        # Wrap in VBox with very tight spacings and simplified headers
        dashboard_layout = widgets.VBox([
            self.title_html,
            widgets.HTML(value='<hr style="margin: 4px 0; border: 0; border-top: 1px solid #e2e8f0;">'),
            controls_box,
            self.progress_bar,
            self.status_html,
            widgets.HTML(value='<hr style="margin: 4px 0; border: 0; border-top: 1px solid #e2e8f0;">'),
            widgets.HBox([
                widgets.VBox([
                    routes_card,
                    self.details_html
                ], layout=widgets.Layout(width='44%')),
                widgets.VBox([
                    widgets.HTML(value='<h3 style="color: #2c3e50; font-family: sans-serif; font-size: 13px; font-weight: bold; margin: 0 0 3px 0;">Interactive Map</h3>'),
                    self.map_html
                ], layout=widgets.Layout(width='54%', margin='0 0 0 2%'))
            ], layout=widgets.Layout(width='100%', margin='2px 0 0 0'))
        ], layout=widgets.Layout(width='100%', padding='5px'))

        return dashboard_layout
