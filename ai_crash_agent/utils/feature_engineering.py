import pandas as pd

EXPECTED_COLUMNS = [
    "crash_speed_limit",
    "num_units",
    "collision_type_grouped_head_on",
    "collision_type_grouped_other",
    "collision_type_grouped_parking_related",
    "collision_type_grouped_rear_end",
    "collision_type_grouped_sideswipe",
    "collision_type_grouped_single_vehicle",
    "collision_type_grouped_turning",
    "day_of_week",
    "time_bucket_midday",
    "time_bucket_morning_rush",
    "time_bucket_non_rush"
]

def process_input(data: dict):

    # Create empty row
    df = pd.DataFrame([[0]*len(EXPECTED_COLUMNS)], columns=EXPECTED_COLUMNS)

    # NUMERIC FEATURES
    df["crash_speed_limit"] = data.get("crash_speed_limit", 35)
    df["num_units"] = data.get("num_units", 1)
    df["day_of_week"] = data.get("day_of_week", 0)

    # COLLISION TYPE (ONE-HOT)
    collision_map = {
        "Head-on": "collision_type_grouped_head_on",
        "Other": "collision_type_grouped_other",
        "Parking": "collision_type_grouped_parking_related",
        "Rear-end": "collision_type_grouped_rear_end",
        "Sideswipe": "collision_type_grouped_sideswipe",
        "Single vehicle": "collision_type_grouped_single_vehicle",
        "Turning": "collision_type_grouped_turning"
    }

    selected_collision = data.get("collision_type", "Other")

    if selected_collision in collision_map:
        df[collision_map[selected_collision]] = 1

    # TIME BUCKETS (from hour)
    hour = data.get("hour", 12)

    if 7 <= hour <= 10:
        df["time_bucket_morning_rush"] = 1
    elif 10 <= hour <= 15:
        df["time_bucket_midday"] = 1
    else:
        df["time_bucket_non_rush"] = 1

    return df