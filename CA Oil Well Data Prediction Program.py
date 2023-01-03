import numpy as np
import geopy.distance
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from datetime import timedelta

pd.set_option('display.max_seq_items', None)

max_range = 1
min_threshold = 10
step_size = 0.01
time_cutoff = timedelta(seconds=0.1)

def readFile(file):
    return pd.read_csv(file, low_memory=False)

def remove_outliers(list):
    array = np.array(list)
    mean = np.mean(array)
    standard_deviation = np.std(array)
    distance_from_mean = abs(array - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    no_outliers = array[not_outlier]

    return no_outliers

def swap_dict(dict):
    swapped = defaultdict(set)
    for k, v in dict.items():
        swapped[v].add(k)
    return swapped

def valid_latitude(latitude):
    if latitude >= -90 and latitude <= 90:
        return True
    return False

def valid_longitude(longitude):
    if longitude >= -180 and longitude <= 180:
        return True
    return False

def merge_csv(all_files):
    # Storing each dataframe into a List
    all_df = []
    for f in all_files:
        df = pd.read_csv(f, sep=',', low_memory=False)
        all_df.append(df)

    # Merge every single dataframe into a giant dataframe
    df = all_df[0]
    for df_temp in all_df[1:]:
        common_columns = set(df.columns).intersection(df_temp.columns)
        df = pd.concat([df, df_temp], axis=0, ignore_index=True, keys=common_columns)

    return df

def percent_error(actual, predicted):
    if actual != 0:
        result = (np.abs(predicted - actual) / actual) * 100
        return result
    else:
        return np.na()

def closest(list, value):
    return list[min(range(len(list)), key = lambda i: abs(list[i] - value))]

def fillValues(file):
    data = readFile(file)

    data['CASING_SIZE'] = pd.to_numeric(data['CASING_SIZE'], errors='coerce')
    data['TrueVerticalDepth'] = pd.to_numeric(data['TrueVerticalDepth'], errors='coerce')
    data['SurfaceLongitude'] = pd.to_numeric(data['SurfaceLongitude'], errors='coerce')
    data['SurfaceLatitude'] = pd.to_numeric(data['SurfaceLatitude'], errors='coerce')

    col_name = "TrueVerticalDepth"
    col_index = data.columns.get_loc(col_name)

    data.insert(loc=col_index + 1, column='PredictedTVDepth', value="")
    data.insert(loc=col_index + 2, column="ErrorPercentagePredicted", value="")
    data.insert(loc=col_index + 3, column='FieldAvgDepth', value="")

    wells_with_field = data[data.Field.notnull()]
    wells_with_data = wells_with_field[wells_with_field.TrueVerticalDepth.notnull()]

    sel_fields = wells_with_data['Field']
    wells_no_data = wells_with_field[wells_with_field.TrueVerticalDepth.isnull()]
    wells_no_data = wells_no_data[wells_no_data['Field'].isin(sel_fields)]

    field_dict = swap_dict(sel_fields.to_dict())

    no_data_dict = wells_no_data.to_dict('records')

    depth_data = wells_with_data['TrueVerticalDepth']
    avg_field_depths = {}

    for field in field_dict:
        wells_same_field = wells_with_data.loc[wells_with_data['Field'] == field]
        avg_field_depth = np.mean(wells_same_field["TrueVerticalDepth"])
        avg_field_depths[field] = avg_field_depth

    data['FieldAvgDepth'] = wells_with_data['Field'].map(avg_field_depths)

    predicted_count = 0
    average_count = 0

    runtimes = []
    wells_iterated_list = []
    for row in tqdm(no_data_dict):
        time_start = datetime.now()
        wells_iterated = 0
        row_index = data.loc[data.API10 == row['API10']].index

        cur_threshold = min_threshold
        in_range_sizes = []

        field = row['Field']

        latitude_1 = row['SurfaceLatitude']
        longitude_1 = row['SurfaceLongitude']

        if valid_latitude(latitude_1) and valid_longitude(longitude_1):
            coords_1 = (latitude_1, longitude_1)

            data.loc[row_index, 'FieldAvgDepth'] = avg_field_depths[field]

            for well_index in field_dict[field]:
                time_elapsed = datetime.now() - time_start
                wells_iterated += 1

                if time_elapsed > time_cutoff:
                    multiples = time_elapsed / time_cutoff
                    if multiples > 0:
                        cur_threshold = min_threshold // (2 * multiples)

                        if cur_threshold < 1:
                            break

                if well_index != row_index:
                    latitude_2 = data.loc[well_index, 'SurfaceLatitude']
                    longitude_2 = data.loc[well_index, 'SurfaceLongitude']

                    if valid_latitude(latitude_2) and valid_longitude(longitude_2):
                        coords_2 = (latitude_2, longitude_2)
                        distance = geopy.distance.geodesic(coords_1, coords_2).miles

                        if distance <= max_range:
                            in_range_sizes.append(depth_data[well_index])

            if len(in_range_sizes) != 0 and len(in_range_sizes) >= cur_threshold:
                no_outliers = remove_outliers(in_range_sizes)
                if len(no_outliers) > 0:
                    average = np.mean(no_outliers)
                else:
                    average = np.mean(in_range_sizes)

                data.loc[row_index, 'PredictedTVDepth'] = average
                predicted_count += 1

        runtime = datetime.now() - time_start
        runtimes.append(runtime)
        wells_iterated_list.append(wells_iterated)

    data['PredictedTVDepth'].fillna(data['FieldAvgDepth'])
    data.to_csv('new_depth_predictions.csv')

    print("Average Runtime:", np.mean(runtimes))
    print("Average Number of Wells Iterated:", np.mean(wells_iterated_list))


    print("Predicted count: ", predicted_count)
    print("Average Count: ", average_count)
    print("Avg ErrorPercentage using Predicted: ", data['ErrorPercentagePredicted'].mean())
    print("Avg ErrorPercentage using FieldAvg: ", data['ErrorPercentageFieldAvg'].mean())

fillValues('new_combined_depth.csv')