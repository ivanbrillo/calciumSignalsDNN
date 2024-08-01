import numpy as np
import pandas as pd
import os


def parse_filename(filename):
    parts = filename.split('_')

    labels = {
        'type': parts[0],
        'date': parts[2],
        'source': parts[3]
    }

    if parts[1] != "#null":
        labels['id'] = parts[1]

    if parts[3] == 'Control':
        labels['stimulus'] = parts[4]
        if len(parts) > 5:
            if "pH" in parts[5]:
                labels['concentration'] = parts[5].split('.')[0]
            else:
                labels['concentration'] = parts[5]
    else:

        if parts[4] != "null":
            labels['frequency_power'] = parts[4]
        labels['activation_time'] = parts[5]

        if parts[6] != "fresh":
            duration_list = parts[6].split(')')

            if len(duration_list) >= 2:
                duration = duration_list[1]
                if "week" in duration:
                    labels['storage_duration'] = int(duration[0]) * 7
                elif "month" in duration:
                    labels['storage_duration'] = int(duration[0]) * 30
                else:
                    labels['storage_duration'] = int(''.join(filter(str.isdigit, duration)))

                temperature = duration_list[0].split('(')[1]
                if temperature == "RT":
                    labels['temperature'] = 25
                else:
                    number = int(''.join(filter(str.isdigit, temperature)))
                    labels['temperature'] = -number if '-' in temperature else number

        else:
            labels['storage_duration'] = 0

        labels['dilution'] = parts[7] if len(parts) > 7 else None

    return labels


def read_excel_to_numpy(file_path):
    df = pd.read_excel(file_path)
    return df.to_numpy()


def create_dataframe(directory='./'):
    data = []

    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):
            labels = parse_filename(filename[:-5])
            file_path = os.path.join(directory, filename)
            time_series_file = read_excel_to_numpy(file_path).T

            for i in range(time_series_file.shape[0]):
                time_series = time_series_file[i]

                if len(time_series) == 1799:
                    time_series = np.append(time_series, [time_series[-1], ])

                if not np.any(time_series is None):
                    data.append({
                        # 'filename': filename,
                        # 'shape_ts': time_series.shape,
                        'time_series': time_series,
                        **labels
                    })

    df = pd.DataFrame(data)
    df['stimulus'] = df['stimulus'].fillna("PAW")
    df['storage_duration'] = df['storage_duration'].fillna(0)

    return df

#
# # Usage
# df = create_dataframe()
#
# pd.set_option('display.max_rows', None)  # Display all rows
# pd.set_option('display.max_columns', None)  # Display all columns
# pd.set_option('display.width', None)  # Adjust display width
# pd.set_option('display.max_colwidth', None)  # Display entire content of columns
#
# print(df)

# # for ts in df['time_series']:
# #     print(ts.shape)
