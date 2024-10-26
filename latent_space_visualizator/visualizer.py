import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, CheckButtons
import pandas as pd
import helper
import data.database_creator as db  # custom module
from lib.latent_space_helper import scale, create_colors, get_colors_list


class Visualizer:

    def __init__(self):
        self.fig, self.ax = helper.setup_frame("Calcium Signals' Latent Space Visualizer")
        self.cid1 = self.ax[0].figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid3 = self.ax[0].figure.canvas.mpl_connect('pick_event', self.onpick)
        # self.cid2 = self.ax[1].figure.canvas.mpl_connect('button_press_event', self.onclick)

        # self.current_dim = 2
        self.show_phy_data = False

        self.full_autoencoder = helper.load_autoencoder(2)
        self.df = db.load_database('./data/dataframe.h5')

        # print(self.df.columns)
        # print(self.df.drop(['filtered', "shape_ts", "time_series", "resampled"], axis=1))

        # result = self.df[self.df['stimulus'] == "PAW"]
        # result = result[result['source'] == "DBD"]
        result = self.df

        nan_clear = result[~result['filtered'].apply(lambda x: np.any(np.isnan(x)))]

        columns = list(nan_clear["stimulus"].unique())
        columns.remove("PAW")

        nan_clear = nan_clear.copy()

        # Convert 'storage_duration' to numeric, fill NaNs, and convert to int then str
        nan_clear['storage_duration'] = (
            pd.to_numeric(nan_clear['storage_duration'], errors='coerce')
            .fillna(0)  # Replace NaNs with 0 if necessary
            .astype(int)  # Convert to int to remove the .0
            .astype(str)  # Convert back to string
        )

        for treatment in columns:
            nan_clear.loc[nan_clear['stimulus'] == treatment, 'activation_time'] = treatment
            nan_clear.loc[nan_clear['stimulus'] == treatment, 'storage_duration'] = treatment
            nan_clear.loc[nan_clear['stimulus'] == treatment, 'frequency_power'] = treatment
            nan_clear.loc[nan_clear['stimulus'] == treatment, 'source'] = treatment

        nan_clear = nan_clear[nan_clear['source'].notna()]
        nan_clear = nan_clear[nan_clear['frequency_power'].notna()]
        nan_clear = nan_clear[nan_clear['storage_duration'].notna()]

        # print(nan_clear)
        # print(nan_clear.columns)

        self.total_df = nan_clear

        nan_clear = nan_clear[nan_clear["stimulus"] == "PAW"]

        self.np_smooth = np.vstack(nan_clear['filtered'].values)
        self.np_arrays = np.vstack(nan_clear['resampled'].values)

        self.df = nan_clear

        # print()
        # self.database, self.np_arrays, self.np_smooth = load_database('../databse.pkl', self.show_phy_data)

        self.pred_val = self.generate_values(self.np_arrays)

        self.current_label = "activation_time"
        self.selected = [False]
        self.coordinate = [0] * 2
        self.scatter = []
        self.scatter_plot()
        self.radio_button = self.create_radiobutton()
        self.picked = (False, 0)
        self.curr_selected = ["PAW"]

    def generate_values(self, values):
        return self.full_autoencoder.encoder.predict(values, verbose=0)[2].T
        # return self.full_autoencoder.encoder.predict(values, verbose=0).T

    def on_button_clicked(self, _):
        self.show_phy_data = not self.show_phy_data
        self.pred_val = self.generate_values(self.np_arrays)
        self.clear_and_plot()

    def create_radiobutton(self):
        rax = plt.axes((0.03, 0.15, 0.2, 0.2))
        radio1 = RadioButtons(rax, list(helper.names.keys()), 0)

        columns = list(self.total_df["stimulus"].unique())
        columns.remove("PAW")

        rax = plt.axes((0.03, 0.35, 0.2, 0.5))
        check2 = CheckButtons(rax, columns)

        # radio2 = RadioButtons(rax, ["2D", "4D"], 0)

        # button_ax = plt.axes((0.05, 0.7, 0.15, 0.12))
        # button = Button(button_ax, 'Show physical data')
        # button.on_clicked(self.on_button_clicked)

        check2.on_clicked(self.change_stim)
        radio1.on_clicked(self.change_type)
        # radio2.on_clicked(self.change_dim)
        return radio1, check2

    def change_stim(self, element):
        if element in self.curr_selected:
            self.curr_selected.remove(element)
        else:
            self.curr_selected.append(element)

        nan_clear = self.total_df[self.total_df["stimulus"].isin(self.curr_selected)]

        self.np_smooth = np.vstack(nan_clear['filtered'].values)
        self.np_arrays = np.vstack(nan_clear['resampled'].values)

        self.df = nan_clear
        self.pred_val = self.generate_values(self.np_arrays)
        self.clear_and_plot()



    def scatter_plot(self):
        colors, self.scatter = create_colors(self.current_label, self.df, self.ax)
        all_colors = get_colors_list(self.df, colors, self.current_label)

        self.scatter.append(self.ax[0].scatter(self.pred_val[0], self.pred_val[1], color=all_colors, picker=True))
        scale(self.ax[0], self.pred_val[:2, :])

    def change_type(self, new_label: str):
        self.current_label = helper.names[new_label]

        # self.df = db.load_database('../data/dataframe.h5')
        # result = self.df[self.df['stimulus'] == "PAW"]

        # nan_clear = result[~result['filtered'].apply(lambda x: np.any(np.isnan(x)))]
        # self.df = nan_clear[~nan_clear[self.current_label].isna()]

        # self.np_smooth = np.vstack(self.df['filtered'].values)
        # self.np_arrays = np.vstack(self.df['resampled'].values)
        # self.pred_val = self.generate_values(self.np_arrays)

        # print(self.np_smooth.shape)

        self.clear_and_plot()

    def clear_and_plot(self):
        for s in self.scatter:
            s.remove()

        self.fig.canvas.draw_idle()
        self.scatter_plot()

    # def change_dim(self, label: str):
    #     # self.current_dim = int(label[0])
    #     self.full_autoencoder = helper.load_autoencoder(self.current_dim)
    #
    #     self.pred_val = self.generate_values(self.np_arrays)
    #     self.clear_and_plot()
    #
    #     self.selected = [False] * (self.current_dim // 2)
    #     self.coordinate = [0] * self.current_dim

    def onclick(self, event):
        if event.inaxes is self.ax[0]:
            self.coordinate[0] = event.xdata
            self.coordinate[1] = event.ydata
            self.selected[0] = True
        # elif event.inaxes is self.ax[1] and self.current_dim == 4:
        #     self.coordinate[2] = event.xdata
        #     self.coordinate[3] = event.ydata
        #     self.selected[1] = True
        self.generate_plt()

    def onpick(self, event):
        self.picked = (True, event.ind[0])

    def generate_plt(self):
        if all(self.selected):

            fig_new, ax_new = plt.subplots(figsize=(6, 6))
            coordinate_np = np.array(self.coordinate).reshape([1, 2])

            if self.picked[0]:
                coordinate_np = np.array([self.pred_val[0][self.picked[1]], self.pred_val[1][self.picked[1]]]).reshape(
                    [1, 2])

                    # print(second_closest)

            values = self.full_autoencoder.decoder(coordinate_np)
            ax_new.plot(values[0], label='Model Reconstructed')

            if self.picked[0]:
                coordinate_np1 = self.np_arrays[self.picked[1]].reshape(1800, )
                coordinate_np2 = self.np_smooth[self.picked[1]].reshape(1800, )

                ax_new.plot(coordinate_np1, label='Original')
                ax_new.plot(coordinate_np2, label='Smoothed')

                row_i = self.df.iloc[self.picked[1]]
                if row_i["stimulus"] != "PAW":
                    coord = coordinate_np.reshape([2, 1])
                    distances = np.linalg.norm(self.pred_val - coord, axis=0)

                    sorted_indices = np.argsort(distances)
                    nearest_index = -1

                    for i in sorted_indices:
                        row_i = self.df.iloc[i]
                        if row_i["stimulus"] == "PAW":
                            nearest_index = i
                            break

                    coordinate_np3 = self.np_smooth[nearest_index].reshape(1800, )
                    ax_new.plot(coordinate_np3, label='Nearest Smoothed')

            ax_new.set_title('Generated Time Series')
            fig_new.canvas.manager.set_window_title("Decoded series")
            ax_new.legend()
            plt.show()

            self.picked = (False, 0)
            self.selected = [False]
