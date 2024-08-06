import math
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

if os.getenv("COLAB_RELEASE_TAG"):
    from calciumSignalsDNN.data.phy_data.data_parser import create_dataframe as phy_creator
    from calciumSignalsDNN.data.single_page.data_parser import create_dataframe as single_creator
    from calciumSignalsDNN.data.multi_page.data_parser import create_dataframe as multi_creator
    from calciumSignalsDNN.data.new_data.data_parser import create_dataframe as new_creator
else:
    from data.phy_data.data_parser import create_dataframe as phy_creator
    from data.single_page.data_parser import create_dataframe as single_creator
    from data.multi_page.data_parser import create_dataframe as multi_creator
    from data.new_data.data_parser import create_dataframe as new_creator



def remove_duplicates(df):
    df['tuple'] = df['time_series'].apply(tuple)
    df_unique = df.drop_duplicates(subset='tuple').drop(columns='tuple')
    df_unique = df_unique.reset_index(drop=True)
    return df_unique


def get_datasets_paw(df):
    result = df[df['stimulus'] == "PAW"]
    train, test = train_test_split(result, test_size=0.2)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train = train[~train['filtered'].apply(lambda x: np.any(np.isnan(x)))]
    test = test[~test['filtered'].apply(lambda x: np.any(np.isnan(x)))]

    return train, test


def load_database(path='dataframe.h5') -> pd.DataFrame:
    return pd.read_hdf(path, key='df')


def resample_array(arr, target_length=1800):
    current_length = len(arr)
    if current_length == target_length:
        return arr

    x_original = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    try:
        f = interpolate.interp1d(x_original, arr)
        return f(x_new)
    except Exception as e:
        print(e)
        print(x_original.shape)
        return None


def atmf(data: np.array, win_size: int = 80, alpha: int = 40) -> np.array:
    data = list(data)
    n = len(data)
    result = [0] * (n - win_size)

    assert win_size > alpha > 0 and alpha % 4 == 0 and win_size % 2 == 0

    for i in range(win_size // 2, n - win_size // 2):
        window = data[i - win_size // 2:i + win_size // 2]
        window.sort()
        result[i - win_size // 2] = sum(window[alpha // 4:win_size - (alpha // 4) * 3]) / (win_size - alpha)

    return np.array([result[0]] * (win_size // 2) + result + [result[-1]] * (win_size // 2))


def grid_plot(original_data, legend=False, labels: list = ['filtered']) -> None:
    n = len(original_data)  # (N/4)x4 grid of subplot
    fig, axes = plt.subplots(ncols=4, nrows=math.ceil(n / 4), layout='constrained',
                             figsize=(3.5 * 4, 3.5 * math.ceil(n / 4)))

    for index, row in original_data.iterrows():
        legend_value = None
        if legend:
            legend_str = f"{row['date']} {row['source']} {row['frequency_power']} {row['activation_time']} {row['storage_duration']}"
            legend_value = legend_str if legend is None else f"{legend_str} {legend}"

        for label in labels:
            axes[int(index / 4)][index % 4].plot(row[label], label=legend_value)

        axes[int(index / 4)][index % 4].legend(loc='lower right', fontsize=7)

    plt.show()


def create_dataframe():
    df_phy = phy_creator('./phy_data')
    df_single = single_creator('./single_page')
    df_multi = multi_creator('./multi_page')
    df_new = new_creator('./new_data')

    result = pd.concat([df_phy, df_single, df_multi, df_new], axis=0, ignore_index=True)
    result = result.reset_index(drop=True)
    result_unique = remove_duplicates(result)
    result_shuffled = result_unique.sample(frac=1, random_state=42).reset_index(drop=True)
    result_shuffled['resampled'] = result_shuffled['time_series'].apply(resample_array)
    result_shuffled['filtered'] = result_shuffled['resampled'].apply(atmf)

    result_shuffled.to_hdf('dataframe.h5', key='df', mode='w')
    return result_shuffled


pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)  # Adjust display width
pd.set_option('display.max_colwidth', None)  # Display entire content of columns

# df = create_dataframe()
# df = load_database()

# train, test = get_datasets_paw(df)
# grid_plot(df[:200], legend=True, labels=['resampled', 'filtered'])
# print(test)
# print(train)
