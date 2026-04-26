import pandas as pd

def process_input(data: dict):
    df = pd.DataFrame([data])

    df['crash_speed_limit_missing'] = df['crash_speed_limit'].isnull().astype(int)
    df['crash_speed_limit'] = df['crash_speed_limit'].fillna(35)

    df['num_units'] = data.get('num_units', 1)

    units = data.get('units', '').lower()
    df['has_pedestrian'] = int('pedestrian' in units)
    df['has_motorcycle'] = int('motorcycle' in units)
    df['has_bicycle'] = int('bicycle' in units)
    df['has_micromobility'] = int('scooter' in units)
    df['has_large_vehicle'] = int('truck' in units)
    df['has_other'] = 0

    df['hour'] = data['hour']
    df['day_of_week'] = data['day_of_week']

    df['road_constr_zone_fl'] = int(data.get('road_construction', 0))
    df['onsys_fl'] = int(data.get('onsys', 1))
    df['private_dr_fl'] = int(data.get('private_drive', 0))

    collision = data.get('collision_type', '').lower()
    df['collision_type_grouped_headon'] = int('head' in collision)
    df['collision_type_grouped_rear_end'] = int('rear' in collision)
    df['collision_type_grouped_angle'] = int('angle' in collision)
    df['collision_type_grouped_other'] = int(
        not any([df['collision_type_grouped_headon'][0],
                 df['collision_type_grouped_rear_end'][0],
                 df['collision_type_grouped_angle'][0]])
    )

    return df
