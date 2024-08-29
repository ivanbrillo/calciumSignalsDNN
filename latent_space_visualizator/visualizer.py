import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
from tensorflow.python.ops.numpy_ops import np_arrays

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
        self.df = db.load_database('../data/dataframe.h5')

        result = self.df[self.df['stimulus'] == "PAW"]
        result = result[result['source'] == "DBD"]

        nan_clear = result[~result['filtered'].apply(lambda x: np.any(np.isnan(x)))]

        self.np_smooth = np.vstack(nan_clear['filtered'].values)
        self.np_arrays = np.vstack(nan_clear['resampled'].values)

        self.df = nan_clear
        # self.database, self.np_arrays, self.np_smooth = load_database('../databse.pkl', self.show_phy_data)

        self.pred_val = self.generate_values(self.np_arrays)

        self.current_label = "activation_time"
        self.selected = [False]
        self.coordinate = [0] * 2
        self.scatter = []
        self.scatter_plot()
        self.radio_button = self.create_radiobutton()
        self.picked = (False, 0)

    def generate_values(self, values):
        # values = values.reshape(1, -1)
        # print(values.shape)
        # print(self.full_autoencoder.encoder.predict(values, verbose=0).T.shape)
        return self.full_autoencoder.encoder.predict(values, verbose=0)[2].T
        # return self.full_autoencoder.encoder.predict(values, verbose=0).T

    def on_button_clicked(self, _):
        self.show_phy_data = not self.show_phy_data
        # self.database, self.np_arrays, self.np_smooth = load_database('../databse.pkl', self.show_phy_data)
        self.pred_val = self.generate_values(self.np_arrays)
        self.clear_and_plot()

    def create_radiobutton(self) -> tuple[RadioButtons, Button]:
        rax = plt.axes((0.05, 0.3, 0.15, 0.2))
        radio1 = RadioButtons(rax, list(helper.names.keys()), 0)

        # rax = plt.axes((0.05, 0.5, 0.15, 0.12))
        # radio2 = RadioButtons(rax, ["2D", "4D"], 0)

        button_ax = plt.axes((0.05, 0.7, 0.15, 0.12))
        button = Button(button_ax, 'Show physical data')
        button.on_clicked(self.on_button_clicked)

        radio1.on_clicked(self.change_type)
        # radio2.on_clicked(self.change_dim)
        return radio1, button

    def scatter_plot(self):
        colors, self.scatter = create_colors(self.current_label, self.df, self.ax)
        all_colors = get_colors_list(self.df, colors, self.current_label)

        self.scatter.append(self.ax[0].scatter(self.pred_val[0], self.pred_val[1], color=all_colors, picker=True))
        scale(self.ax[0], self.pred_val[:2, :])

        # if self.current_dim == 4:
        #     self.scatter.append(self.ax[1].scatter(self.pred_val[2], self.pred_val[3], color=all_colors, picker=True))
        #
        #     scale(self.ax[1], self.pred_val[2:, :])

    def change_type(self, new_label: str):
        self.current_label = helper.names[new_label]

        self.df = db.load_database('../data/dataframe.h5')
        result = self.df[self.df['stimulus'] == "PAW"]

        nan_clear = result[~result['filtered'].apply(lambda x: np.any(np.isnan(x)))]
        self.df = nan_clear[~nan_clear[self.current_label].isna()]

        self.np_smooth = np.vstack(self.df['filtered'].values)
        self.np_arrays = np.vstack(self.df['resampled'].values)
        self.pred_val = self.generate_values(self.np_arrays)

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

            values = self.full_autoencoder.decoder(coordinate_np)

            ax_new.plot(values[0], label='Reconstructed')

            if self.picked[0]:
                coordinate_np1 = self.np_arrays[self.picked[1]].reshape(1800, )
                coordinate_np2 = self.np_smooth[self.picked[1]].reshape(1800, )

                ax_new.plot(coordinate_np1, label='Original')
                ax_new.plot(coordinate_np2, label='Smoothed')

            ax_new.set_title('Generated Time Series')
            fig_new.canvas.manager.set_window_title("Decoded series")
            ax_new.legend()
            plt.show()

            self.picked = (False, 0)
            self.selected = [False]
