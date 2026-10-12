import numpy as np
import pandas as pd

class RobustJourneyPlanner:
    """
    Reverse CSA candidate-route planner.

    Pre-confidence version:
        - Generates timetable-feasible candidate routes.
        - Avoids artificial vehicle-hopping by creating direct same-trip ride connections.
        - Keeps confidence = 1.0 as placeholder.
        - Extracts transfer/final-arrival events for the future delay model.
        - Accepts stop IDs or stop names as input.
        - Expands general/parent stops to matching variants when useful.
    """

    def __init__(self):
        self.stops = None
        self.stop_names = {}
        self.walk_by_stop = {}
        self.conn_data = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stop_key(stop_id):
        return str(stop_id)

    @staticmethod
    def _bpuic(stop_id) -> int:
        return int(str(stop_id).split(":")[0])

    @staticmethod
    def _to_seconds(hhmm_ss) -> int:
        """Convert HH:MM or HH:MM:SS to seconds from midnight."""
        if pd.isna(hhmm_ss):
            return 0

        parts = list(map(int, str(hhmm_ss).split(":")))

        if len(parts) == 2:
            return parts[0] * 3600 + parts[1] * 60

        return parts[0] * 3600 + parts[1] * 60 + parts[2]

    @staticmethod
    def _to_time_str(seconds: int) -> str:
        """Convert seconds from midnight to HH:MM:SS."""
        if seconds < 0:
            return "-INF"

        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60

        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _spark_day_of_week(day: str) -> int:
        """Spark dayofweek convention: Sunday=1, Monday=2, ..., Saturday=7."""
        return {
            "sunday": 1,
            "monday": 2,
            "tuesday": 3,
            "wednesday": 4,
            "thursday": 5,
            "friday": 6,
            "saturday": 7,
        }[str(day).lower()]

    @staticmethod
    def _day_type_from_spark_day(day_of_week: int) -> str:
        return "weekend" if int(day_of_week) in (1, 7) else "weekday"

    @staticmethod
    def _time_bucket_15min(seconds: int) -> int:
        return int(((int(seconds) % (24 * 3600)) // 60) // 15)

    @staticmethod
    def _hour_of_day(seconds: int) -> int:
        return int((int(seconds) % (24 * 3600)) // 3600)

    @staticmethod
    def _hour_of_week(day_of_week: int, hour: int) -> int:
        return (int(day_of_week) - 1) * 24 + int(hour)

    def _name(self, stop_id):
        stop_id = self._stop_key(stop_id)
        return self.stop_names.get(stop_id, stop_id)

    def _simplify_stop_name(self, name):
        """
        Simplify stop names for matching.

        Examples:
            Lausanne, Provence       -> provence
            Lausanne, Provence nord  -> provence nord
            Lausanne-Flon            -> flon
        """
        name = str(name).lower().strip()
        name = name.replace("lausanne, ", "")
        name = name.replace("lausanne-", "")
        return name

    def _resolve_stop_input(self, stop_input):
        """
        Accept either:
            - exact stop_id, e.g. '8592086'
            - exact stop name, e.g. 'Lausanne, Ouchy-Olympique'
            - partial stop name, e.g. 'Ouchy-Olympique'

        Returns one stop_id string.
        """
        stop_input = str(stop_input)

        if self.stops is None:
            return stop_input

        # 1. Exact stop_id match.
        id_matches = self.stops[
            self.stops["stop_id"].astype(str) == stop_input
        ]

        if len(id_matches) > 0:
            return str(id_matches.iloc[0]["stop_id"])

        # 2. Exact stop_name match.
        exact_name_matches = self.stops[
            self.stops["stop_name"].astype(str).str.lower() == stop_input.lower()
        ]

        if len(exact_name_matches) > 0:
            return str(exact_name_matches.iloc[0]["stop_id"])

        # 3. Partial stop_name match.
        partial_matches = self.stops[
            self.stops["stop_name"]
            .astype(str)
            .str.lower()
            .str.contains(stop_input.lower(), na=False)
        ]

        if len(partial_matches) > 0:
            return str(partial_matches.iloc[0]["stop_id"])

        raise ValueError(f"Could not resolve stop input: {stop_input}")

    def _stop_variants(self, stop_id):
        """
        Return possible stop variants for a stop_id.

        This helps when the input is a general stop such as:
            Lausanne, Provence

        and the timetable actually uses:
            Lausanne, Provence nord
            Lausanne, Provence sud
        """
        stop_id = self._stop_key(stop_id)

        if self.stops is None:
            return [stop_id]

        stop_ids = set(self.stops["stop_id"].astype(str))
        variants = {stop_id}

        # 1. Same ID/platform family.
        base = stop_id.split(":")[0]

        for sid in stop_ids:
            if sid.split(":")[0] == base:
                variants.add(sid)

        # 2. Name-based matching.
        if stop_id in stop_ids:
            exact_rows = self.stops[self.stops["stop_id"].astype(str) == stop_id]

            if len(exact_rows) > 0 and "stop_name" in exact_rows.columns:
                name = exact_rows.iloc[0]["stop_name"]
                simple_name = self._simplify_stop_name(name)

                name_matches = self.stops[
                    self.stops["stop_name"]
                    .astype(str)
                    .apply(self._simplify_stop_name)
                    .str.startswith(simple_name, na=False)
                ]

                variants.update(name_matches["stop_id"].astype(str).tolist())

        variants = [v for v in variants if v in stop_ids]

        return sorted(variants)

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------

    def prepare(
        self,
        stops_df,
        walk_df,
        stop_times_df,
        max_same_trip_skip=30,
        max_walk_neighbors=12,
    ):
        """
        Prepare routing structures.

        Parameters
        ----------
        stops_df:
            Must contain stop_id and ideally stop_name.

        walk_df:
            Must contain a_stop_id, b_stop_id, distance.

        stop_times_df:
            Must contain:
                trip_id, stop_id, stop_sequence,
                arrival_time, departure_time,
                monday, tuesday, wednesday, thursday,
                friday, saturday, sunday.

        max_same_trip_skip:
            How many future stops on the same trip we connect directly.
            Use 30 for metro/train lines so the planner can stay on the vehicle.

        max_walk_neighbors:
            Maximum number of nearest walking edges kept per stop. This keeps
            dense walk graphs from dominating query time when max_walk_m is high.
        """
        print("Loading routing structures...")

        self.stops = stops_df.copy()
        self.stops["stop_id"] = self.stops["stop_id"].apply(self._stop_key)

        if "stop_name" in self.stops.columns:
            self.stop_names = dict(
                zip(
                    self.stops["stop_id"].astype(str),
                    self.stops["stop_name"].astype(str),
                )
            )
        else:
            self.stop_names = {}

        self.walk_by_stop = {}

        for row in walk_df.itertuples(index=False):
            a = self._stop_key(row.a_stop_id)
            b = self._stop_key(row.b_stop_id)
            d = float(row.distance)

            self.walk_by_stop.setdefault(a, []).append((b, d))
            self.walk_by_stop.setdefault(b, []).append((a, d))

        if max_walk_neighbors is not None:
            for stop_id, edges in list(self.walk_by_stop.items()):
                self.walk_by_stop[stop_id] = sorted(edges, key=lambda x: x[1])[: int(max_walk_neighbors)]

        st = stop_times_df.copy()
        st["stop_id"] = st["stop_id"].apply(self._stop_key)
        st["arrival_time"] = st["arrival_time"].apply(self._to_seconds)
        st["departure_time"] = st["departure_time"].apply(self._to_seconds)

        st = st.sort_values(["trip_id", "stop_sequence"])

        optional_defaults = {
            "operator_id": "UNKNOWN",
            "line_id": "UNKNOWN",
            "line_text": "UNKNOWN",
            "transport_clean": "UNKNOWN",
            "product_id_clean": "UNKNOWN",
        }

        for col, default in optional_defaults.items():
            if col not in st.columns:
                st[col] = default
            else:
                st[col] = st[col].fillna(default).astype(str)

        day_cols = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]

        connection_rows = []

        for trip_id, group in st.groupby("trip_id", sort=False):
            group = group.sort_values("stop_sequence").reset_index(drop=True)
            rows = list(group.itertuples(index=False))

            # Direct ride connections from each stop to later stops on the same trip.
            # This prevents fake stop-by-stop vehicle hopping.
            for i in range(len(rows) - 1):
                from_row = rows[i]
                max_j = min(len(rows), i + 1 + max_same_trip_skip)

                for j in range(i + 1, max_j):
                    to_row = rows[j]
                    arrival_segment_from_row = rows[j - 1]

                    dep_t = int(from_row.departure_time)
                    arr_t = int(to_row.arrival_time)

                    if arr_t < dep_t:
                        continue

                    row = {
                        "trip_id": trip_id,
                        "from_stop": from_row.stop_id,
                        "to_stop": to_row.stop_id,
                        "departure_time": dep_t,
                        "arrival_time": arr_t,
                        "from_sequence": int(from_row.stop_sequence),
                        "to_sequence": int(to_row.stop_sequence),
                        "arrival_segment_from_stop": arrival_segment_from_row.stop_id,
                        "arrival_segment_dep_time": int(arrival_segment_from_row.departure_time),
                        "arrival_segment_from_sequence": int(arrival_segment_from_row.stop_sequence),
                        "n_stops_in_trip": int(len(rows)),
                        "operator_id": from_row.operator_id,
                        "line_id": from_row.line_id,
                        "line_text": from_row.line_text,
                        "transport_clean": from_row.transport_clean,
                        "product_id_clean": from_row.product_id_clean,
                    }

                    for col in day_cols:
                        row[col] = bool(getattr(from_row, col))

                    connection_rows.append(row)

        con = pd.DataFrame(connection_rows)

        if con.empty:
            raise ValueError("No valid transit connections were created.")

        con = con.sort_values("departure_time", ascending=False).reset_index(drop=True)

        self.conn_data = {
            "from_stop": con["from_stop"].values,
            "to_stop": con["to_stop"].values,
            "dep_time": con["departure_time"].values.astype(np.int32),
            "arr_time": con["arrival_time"].values.astype(np.int32),
            "trip_id": con["trip_id"].values,
            "from_sequence": con["from_sequence"].values.astype(np.int32),
            "to_sequence": con["to_sequence"].values.astype(np.int32),
            "arrival_segment_from_stop": con["arrival_segment_from_stop"].values,
            "arrival_segment_dep_time": con["arrival_segment_dep_time"].values.astype(np.int32),
            "arrival_segment_from_sequence": con["arrival_segment_from_sequence"].values.astype(np.int32),
            "n_stops_in_trip": con["n_stops_in_trip"].values.astype(np.int32),
            "operator_id": con["operator_id"].values,
            "line_id": con["line_id"].values,
            "line_text": con["line_text"].values,
            "transport_clean": con["transport_clean"].values,
            "product_id_clean": con["product_id_clean"].values,
            "days_matrix": con[day_cols].values.astype(bool),
        }

        print(
            f"Preparation complete: "
            f"{len(con)} direct same-trip ride connections vectorized."
        )

    # ------------------------------------------------------------------
    # Label handling
    # ------------------------------------------------------------------

    def _add_label(self, labels, stop_id, new_label, k=12):
        """
        Add a label to a stop.

        Candidate-generation version:
            - Keeps several timetable-feasible alternatives.
            - Only removes exact duplicates.
            - Does not remove earlier routes just because a later one exists.
        """
        stop_id = self._stop_key(stop_id)
        existing = labels.get(stop_id, [])

        for lab in existing:
            same_time = lab["time"] == new_label["time"]
            same_walk = abs(lab["walk_m"] - new_label["walk_m"]) < 1e-6
            same_legs = lab["transit_legs"] == new_label["transit_legs"]

            if same_time and same_walk and same_legs:
                return False

        existing.append(new_label)

        existing = sorted(
            existing,
            key=lambda lab: (-lab["time"], lab["transit_legs"], lab["walk_m"]),
        )

        kept = existing[:k]
        labels[stop_id] = kept
        return any(lab is new_label for lab in kept)

    # ------------------------------------------------------------------
    # Main routing
    # ------------------------------------------------------------------

    def route_reverse(
        self,
        start_id,
        end_id,
        arrival_time="18:00",
        day="monday",
        max_walk_m=500,
        max_routes=10,
        max_labels_per_stop=12,
        min_transfer_seconds=120,
        max_transit_legs=6,
        min_final_slack_seconds=60,
        max_search_seconds=4 * 3600,
        expand_stop_variants=True,
    ):
        """
        Reverse CSA candidate route generator.

        start_id and end_id may be either stop IDs or stop names.

        Returns timetable-feasible candidates:
            - sorted by latest departure first
            - with placeholder confidence = 1.0
            - with events_for_model for later confidence scoring
        """
        if self.stops is None or not self.conn_data:
            raise ValueError("Call prepare(...) before route_reverse(...).")

        days_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        day = day.lower()

        if day not in days_map:
            raise ValueError(f"Unknown day: {day}")

        start_id = self._resolve_stop_input(start_id)
        end_id = self._resolve_stop_input(end_id)

        day_idx = days_map[day]
        target_t = self._to_seconds(arrival_time)
        routing_target_t = target_t - min_final_slack_seconds

        walking_speed_m_per_sec = 50 / 60

        labels = {
            self._stop_key(s): []
            for s in self.stops["stop_id"].unique()
        }

        # If the traveller walks from the requested origin to a nearby stop,
        # this should appear explicitly as a WALK step.
        start_variants = [start_id] if start_id in labels else []

        # It is okay to expand the destination, because arriving at a platform/variant
        # of the destination should still count as reaching the destination.
        if expand_stop_variants:
            end_variants = [
                s for s in self._stop_variants(end_id)
                if s in labels
            ]
        else:
            end_variants = [end_id] if end_id in labels else []

        if not start_variants:
            start_variants = [start_id]

        if not end_variants:
            end_variants = [end_id]

        # Initialize all destination variants.
        for end_variant in end_variants:
            if end_variant not in labels:
                continue

            labels[end_variant].append(
                {
                    "time": int(routing_target_t),
                    "walk_m": 0.0,
                    "transit_legs": 0,
                    "parent": None,
                    "is_destination": True,
                }
            )

        def relax_backward_walk_from(source_stop, seed_labels=None):
            """
            Add one walking step into an already reachable stop.

            The walk graph already contains direct edges for all stop pairs
            within max_walk_m, so one-hop relaxation is enough and avoids
            slow chains of consecutive walking labels.
            """
            v = self._stop_key(source_stop)

            source_labels = list(labels.get(v, [])) if seed_labels is None else list(seed_labels)

            for u, dist in self.walk_by_stop.get(v, []):
                u = self._stop_key(u)
                dist = float(dist)

                if dist > max_walk_m:
                    continue

                walk_time = int(dist / walking_speed_m_per_sec)

                for lab_v in source_labels:
                    new_walk_m = lab_v["walk_m"] + dist

                    if new_walk_m > max_walk_m:
                        continue

                    cand_t = int(lab_v["time"] - walk_time)

                    new_label = {
                        "time": cand_t,
                        "walk_m": new_walk_m,
                        "transit_legs": lab_v["transit_legs"],
                        "parent": {
                            "type": "WALK",
                            "from": u,
                            "to": v,
                            "dep_time": cand_t,
                            "arr_time": int(lab_v["time"]),
                            "distance_m": dist,
                            "next_label": lab_v,
                        },
                        "is_destination": False,
                    }

                    self._add_label(
                        labels,
                        u,
                        new_label,
                        k=max_labels_per_stop,
                    )

        # Initial walking into every destination variant.
        for end_variant in end_variants:
            if end_variant in labels:
                relax_backward_walk_from(end_variant)

        valid_mask = (
            self.conn_data["days_matrix"][:, day_idx]
            & (self.conn_data["arr_time"] <= routing_target_t)
            & (self.conn_data["dep_time"] >= int(target_t - max_search_seconds))
        )

        indices = np.where(valid_mask)[0]

        for i in indices:
            u = self._stop_key(self.conn_data["from_stop"][i])
            v = self._stop_key(self.conn_data["to_stop"][i])
            dep_t = int(self.conn_data["dep_time"][i])
            arr_t = int(self.conn_data["arr_time"][i])
            trip = self.conn_data["trip_id"][i]

            from_sequence = int(self.conn_data["from_sequence"][i])
            to_sequence = int(self.conn_data["to_sequence"][i])
            arrival_segment_from_stop = self._stop_key(self.conn_data["arrival_segment_from_stop"][i])
            arrival_segment_dep_time = int(self.conn_data["arrival_segment_dep_time"][i])
            arrival_segment_from_sequence = int(self.conn_data["arrival_segment_from_sequence"][i])
            n_stops_in_trip = int(self.conn_data["n_stops_in_trip"][i])

            for lab_v in list(labels.get(v, [])):
                required_departure_from_v = int(lab_v["time"])

                if arr_t > required_departure_from_v:
                    continue

                slack = required_departure_from_v - arr_t

                if not lab_v.get("is_destination", False) and slack < min_transfer_seconds:
                    continue

                new_transit_legs = lab_v["transit_legs"] + 1

                if new_transit_legs > max_transit_legs:
                    continue

                new_label = {
                    "time": dep_t,
                    "walk_m": lab_v["walk_m"],
                    "transit_legs": new_transit_legs,
                    "parent": {
                        "type": "TRANSIT",
                        "from": u,
                        "to": v,
                        "dep_time": dep_t,
                        "arr_time": arr_t,
                        "trip_id": trip,
                        "from_sequence": from_sequence,
                        "to_sequence": to_sequence,
                        "arrival_segment_from_stop": arrival_segment_from_stop,
                        "arrival_segment_dep_time": arrival_segment_dep_time,
                        "arrival_segment_from_sequence": arrival_segment_from_sequence,
                        "n_stops_in_trip": n_stops_in_trip,
                        "operator_id": self.conn_data["operator_id"][i],
                        "line_id": self.conn_data["line_id"][i],
                        "line_text": self.conn_data["line_text"][i],
                        "transport_clean": self.conn_data["transport_clean"][i],
                        "product_id_clean": self.conn_data["product_id_clean"][i],
                        "slack_after_arrival": slack,
                        "next_label": lab_v,
                    },
                    "is_destination": False,
                }

                added = self._add_label(
                    labels,
                    u,
                    new_label,
                    k=max_labels_per_stop,
                )

                if added and max_walk_m > 0 and new_transit_legs < max_transit_legs and new_label["walk_m"] < max_walk_m:
                    relax_backward_walk_from(u, seed_labels=[new_label])

        # Collect labels from all start variants.
        start_labels = []

        for start_variant in start_variants:
            start_labels.extend(labels.get(start_variant, []))

        if not start_labels:
            print("No timetable-feasible route found.")
            return []

        start_labels = sorted(
            start_labels,
            key=lambda lab: (-lab["time"], lab["transit_legs"], lab["walk_m"]),
        )

        routes = []

        for lab in start_labels:
            steps = self._reconstruct_from_label(lab)

            if not steps:
                continue

            events = self._extract_confidence_events(
                steps,
                arrival_deadline_seconds=target_t,
                day=day,
                min_transfer_seconds=min_transfer_seconds,
            )

            has_bad_transfer = any(
                event["event_type"] == "transfer"
                and event["slack_seconds"] <= 0
                for event in events
            )

            if has_bad_transfer:
                continue

            has_walk_loop = False

            for i in range(len(steps) - 1):
                a = steps[i]
                b = steps[i + 1]

                if (
                    a["type"] == "WALK"
                    and b["type"] == "WALK"
                    and a["from"] == b["to"]
                    and a["to"] == b["from"]
                ):
                    has_walk_loop = True
                    break

            if has_walk_loop:
                continue

            has_multiple_consecutive_walks = any(
                steps[i]["type"] == "WALK" and steps[i + 1]["type"] == "WALK"
                for i in range(len(steps) - 1)
            )

            if has_multiple_consecutive_walks:
                continue

            stop_sequence = [steps[0]["from"]]

            for step in steps:
                stop_sequence.append(step["to"])

            if len(stop_sequence) != len(set(stop_sequence)):
                continue

            routes.append(
                {
                    "departure_time": self._to_time_str(int(lab["time"])),
                    "departure_seconds": int(lab["time"]),
                    "arrival_deadline": self._to_time_str(int(target_t)),
                    "arrival_deadline_seconds": int(target_t),
                    "confidence": 1.0,
                    "walk_m": float(lab["walk_m"]),
                    "transit_legs": int(
                        sum(1 for step in steps if step["type"] == "TRANSIT")
                    ),
                    "steps": steps,
                    "events_for_model": events,
                }
            )

            if len(routes) >= max_routes:
                break

        routes = sorted(
            routes,
            key=lambda r: (-r["departure_seconds"], r["transit_legs"], r["walk_m"]),
        )

        return routes[:max_routes]

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_from_label(self, label):
        steps = []
        current_label = label

        while current_label is not None and current_label["parent"] is not None:
            step = current_label["parent"]

            clean_step = {
                key: value
                for key, value in step.items()
                if key != "next_label"
            }

            clean_step["dep_time_str"] = self._to_time_str(clean_step["dep_time"])
            clean_step["arr_time_str"] = self._to_time_str(clean_step["arr_time"])

            steps.append(clean_step)
            current_label = step["next_label"]

        return steps

    # ------------------------------------------------------------------
    # Events for future confidence model
    # ------------------------------------------------------------------

    def _extract_confidence_events(
        self,
        steps,
        arrival_deadline_seconds,
        day=None,
        min_transfer_seconds=120,
    ):
        events = []

        if not steps:
            return events

        day_of_week = self._spark_day_of_week(day) if day is not None else None
        day_type = (
            self._day_type_from_spark_day(day_of_week)
            if day_of_week is not None
            else None
        )

        def time_context(seconds):
            hour = self._hour_of_day(seconds)
            context = {
                "hour": hour,
                "time_bucket_15min": self._time_bucket_15min(seconds),
            }

            if day_of_week is not None:
                context.update(
                    {
                        "day_of_week": day_of_week,
                        "day_type": day_type,
                        "hour_of_week": self._hour_of_week(day_of_week, hour),
                    }
                )

            return context

        def segment_context(step):
            arrival_segment_from_stop = step.get(
                "arrival_segment_from_stop",
                step["from"],
            )
            arrival_segment_dep_time = int(step.get(
                "arrival_segment_dep_time",
                step["dep_time"],
            ))
            segment_sequence_proxy = int(step.get(
                "arrival_segment_from_sequence",
                max(1, int(step.get("to_sequence", 2)) - 1),
            ))
            n_segments_in_trip = max(
                1,
                int(step.get("n_stops_in_trip", int(step.get("to_sequence", 2)))) - 1,
            )

            return {
                "ride_from_bpuic": self._bpuic(step["from"]),
                "from_bpuic": self._bpuic(arrival_segment_from_stop),
                "to_bpuic": self._bpuic(step["to"]),
                "bpuic": self._bpuic(step["to"]),
                "scheduled_segment_departure": arrival_segment_dep_time,
                "scheduled_travel_time_min": max(
                    0.0,
                    (int(step["arr_time"]) - arrival_segment_dep_time) / 60.0,
                ),
                "segment_sequence_proxy": segment_sequence_proxy,
                "n_segments_in_trip": n_segments_in_trip,
                "trip_progress": float(segment_sequence_proxy) / float(n_segments_in_trip),
                "operator_id": str(step.get("operator_id", "UNKNOWN")),
                "line_id": str(step.get("line_id", "UNKNOWN")),
                "line_text": str(step.get("line_text", "UNKNOWN")),
                "transport_clean": str(step.get("transport_clean", "UNKNOWN")),
                "product_id_clean": str(step.get("product_id_clean", "UNKNOWN")),
            }

        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]

            if current_step["type"] != "TRANSIT":
                continue

            # If the route ends with walking, the final-arrival event handles the risk.
            if i == len(steps) - 2 and next_step["type"] == "WALK":
                continue

            slack = int(
                next_step["dep_time"]
                - current_step["arr_time"]
                - min_transfer_seconds
            )

            events.append(
                {
                    "event_type": "transfer",
                    "trip_id": current_step.get("trip_id"),
                    "stop_id": current_step["to"],
                    "scheduled_arrival": int(current_step["arr_time"]),
                    "scheduled_arrival_str": current_step["arr_time_str"],
                    "required_next_departure": int(next_step["dep_time"]),
                    "required_next_departure_str": next_step["dep_time_str"],
                    "min_transfer_seconds": int(min_transfer_seconds),
                    "slack_seconds": int(slack),
                    "usable_slack_min": float(slack) / 60.0,
                    **segment_context(current_step),
                    **time_context(current_step["arr_time"]),
                }
            )

        if steps[-1]["type"] == "WALK":
            last_walk = steps[-1]
            previous_transit = None

            for step in reversed(steps[:-1]):
                if step["type"] == "TRANSIT":
                    previous_transit = step
                    break

            if previous_transit is not None:
                final_slack = int(arrival_deadline_seconds - last_walk["arr_time"])

                events.append(
                    {
                        "event_type": "final_arrival",
                        "trip_id": previous_transit.get("trip_id"),
                        "stop_id": previous_transit["to"],
                        "scheduled_arrival": int(previous_transit["arr_time"]),
                        "scheduled_arrival_str": previous_transit["arr_time_str"],
                        "scheduled_route_arrival": int(last_walk["arr_time"]),
                        "scheduled_route_arrival_str": last_walk["arr_time_str"],
                        "arrival_deadline": int(arrival_deadline_seconds),
                        "arrival_deadline_str": self._to_time_str(arrival_deadline_seconds),
                        "slack_seconds": int(final_slack),
                        "usable_slack_min": float(final_slack) / 60.0,
                        **segment_context(previous_transit),
                        **time_context(previous_transit["arr_time"]),
                    }
                )

        else:
            final_step = steps[-1]
            final_arrival = int(final_step["arr_time"])
            final_slack = int(arrival_deadline_seconds - final_arrival)

            events.append(
                {
                    "event_type": "final_arrival",
                    "trip_id": final_step.get("trip_id"),
                    "stop_id": final_step["to"],
                    "scheduled_arrival": final_arrival,
                    "scheduled_arrival_str": final_step["arr_time_str"],
                    "scheduled_route_arrival": final_arrival,
                    "scheduled_route_arrival_str": final_step["arr_time_str"],
                    "arrival_deadline": int(arrival_deadline_seconds),
                    "arrival_deadline_str": self._to_time_str(arrival_deadline_seconds),
                    "slack_seconds": int(final_slack),
                    "usable_slack_min": float(final_slack) / 60.0,
                    **segment_context(final_step),
                    **time_context(final_step["arr_time"]),
                }
            )

        return events

    def score_routes_with_confidence_model(
        self,
        routes,
        confidence_function,
        min_confidence=0.0,
    ):
        """
        Score candidate routes using a teammate-provided confidence model.

        confidence_function(event) must return:
            P(success for this event), usually from the reduced delay model
            plus the validation residual CDF lookup.
        """
        scored_routes = []

        for route in routes:
            confidence = 1.0

            for event in route["events_for_model"]:
                p = float(confidence_function(event))
                confidence *= p

            new_route = route.copy()
            new_route["confidence"] = float(confidence)

            if confidence >= min_confidence:
                scored_routes.append(new_route)

        scored_routes = sorted(
            scored_routes,
            key=lambda r: (-r["departure_seconds"], -r["confidence"], r["walk_m"]),
        )

        return scored_routes

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_routes(self, routes, show_model_events=False):
        if not routes:
            print("No routes to display.")
            return

        for idx, route in enumerate(routes, start=1):
            print(f"\n=== ROUTE {idx} ===")
            print(f"Departure time: {route['departure_time']}")
            print(f"Arrival deadline: {route['arrival_deadline']}")
            print(f"Current confidence: {route['confidence']:.3f}")
            print(f"Walking distance: {route['walk_m']:.1f} m")
            print(f"Transit legs: {route['transit_legs']}")
            print()

            for step in route["steps"]:
                if step["type"] == "TRANSIT":
                    print(
                        f"[{step['dep_time_str']}] BOARD trip '{step['trip_id']}' "
                        f"at {self._name(step['from'])}"
                    )
                    print(
                        f"[{step['arr_time_str']}] ALIGHT at {self._name(step['to'])}"
                    )

                elif step["type"] == "WALK":
                    from_name = self._name(step["from"])
                    to_name = self._name(step["to"])

                    if from_name == to_name:
                        print(
                            f"[{step['dep_time_str']}] WALK within {from_name} "
                            f"({step['distance_m']:.0f} m)"
                        )
                    else:
                        print(
                            f"[{step['dep_time_str']}] WALK from {from_name} "
                            f"to {to_name} "
                            f"({step['distance_m']:.0f} m)"
                        )

                print("   |")

            actual_arrival = route["steps"][-1]["arr_time_str"]
            print(f"[{actual_arrival}] ARRIVE")
            print(f"Deadline: {route['arrival_deadline']}")

            if show_model_events:
                print("\nEvents to send to the delay model later:")

                for event in route["events_for_model"]:
                    print(
                        f"  - {event['event_type']} at stop {event['stop_id']} "
                        f"(bpuic={event['bpuic']}), "
                        f"scheduled arrival {event['scheduled_arrival_str']}, "
                        f"slack = {event['slack_seconds']} sec"
                    )

if __name__ == "__main__":
    pass
