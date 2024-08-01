import numpy as np
import pandas as pd
import os


def parse_filename(filename):
    parts = filename.split('_')
    labels = {'source': "Control", 'stimulus': parts[4], 'concentration': parts[3], 'storage_duration': 0}
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
    return df

# # Usage
# df = create_dataframe()
#
# pd.set_option('display.max_rows', None)  # Display all rows
# pd.set_option('display.max_columns', None)  # Display all columns
# pd.set_option('display.width', None)  # Adjust display width
# pd.set_option('display.max_colwidth', None)  # Display entire content of columns
#
# print(df)

# for ts in df['time_series']:
#     print(ts.shape)
