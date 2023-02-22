#!/usr/bin/env python
# Copyright (C) 2023 - All rights reserved.
# Author Contact Info: Kourosh.Darvish@gmail.com
# SPDX-License-Identifier: BSD-3-Clause
#

import copy
import datetime
import math
import os
import sys
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yarp
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot

from kindyn_prediction.utils.dataset_utils import current_milli_time

# parameters:
number_motion_output = 66
labels = ["None", "Rotating", "Standing", "Walking"]
number_categories = len(labels)

# yarp related code
if not yarp.Network.checkNetwork():
    print("[main] Unable to find YARP network")
yarp.Network.init()
rf = yarp.ResourceFinder()
rf.setDefaultContext("myContext")
rf.setDefaultConfigFile("default.ini")

action_prediction_port = yarp.BufferedPortVector()
action_prediction_port.open("/onlineAnimation/actionRecognition:i")
is_connected_human_action_prediction = yarp.Network.connect(
    "/test_moe/actionRecognition:o", "/onlineAnimation/actionRecognition:i"
)
print(f"action prediction port is connected: {is_connected_human_action_prediction}")
yarp.delay(0.5)


class PlotInferenceResults:
    def __init__(self):
        # related to figure
        font = {"size": 15}
        matplotlib.rc("font", **font)

        self.xmin = 0.0
        self.xmax = 6.5
        self.plot_front_time = 1.2
        self.f0 = figure(num=0, figsize=(8, 3.0))  # , dpi=100)
        self.ax01 = self.f0.subplots()  # 2grid((1, 1), (0, 0))
        self.ax01.set_ylim(-0.1, 1.1)
        self.ax01.set_xlim(self.xmin, self.xmax)
        self.ax01.grid(False)
        self.t = np.zeros(0)
        self.t0 = current_milli_time() / 1000.0  # seconds

        self.t_prediction = np.zeros(0)
        # action 0
        self.prediction_now0 = np.zeros(0)
        self.action_predictions0 = np.zeros(0)
        (self.p1,) = self.ax01.plot(self.t, self.prediction_now0, "k-", linewidth=5, label=f"{labels[0]}")
        (self.p2,) = self.ax01.plot(
            self.t_prediction, self.action_predictions0, "o", color="k", markersize=4, alpha=0.05
        )

        # action 1
        self.prediction_now1 = np.zeros(0)
        self.action_predictions1 = np.zeros(0)
        (self.p3,) = self.ax01.plot(self.t, self.prediction_now1, "b-", linewidth=5, label=f"{labels[1]}")
        (self.p4,) = self.ax01.plot(
            self.t_prediction, self.action_predictions1, "o", color="b", markersize=4, alpha=0.05
        )

        # action 2
        self.prediction_now2 = np.zeros(0)
        self.action_predictions2 = np.zeros(0)
        (self.p5,) = self.ax01.plot(self.t, self.prediction_now2, "r-", linewidth=5, label=f"{labels[2]}")
        (self.p6,) = self.ax01.plot(
            self.t_prediction, self.action_predictions2, "o", color="r", markersize=4, alpha=0.05
        )

        # action 3
        self.prediction_now3 = np.zeros(0)
        self.action_predictions3 = np.zeros(0)
        (self.p7,) = self.ax01.plot(self.t, self.prediction_now3, "g-", linewidth=5, label=f"{labels[3]}")
        (self.p8,) = self.ax01.plot(
            self.t_prediction, self.action_predictions3, "o", color="g", markersize=4, alpha=0.05
        )

        # related to the data
        self.timer = current_milli_time()
        self.counter = 0
        self.time_length = 100
        self.human_kin_dyn_data = []

        self.prediction_horizon = 25
        self.time_step = 0.04
        self.output_size = 4

        return

    def animate(self, dummy):
        # read human current data:
        # data manipulation
        print(f"timer: {current_milli_time() - self.timer}")
        self.timer = current_milli_time()
        time_now = (current_milli_time() / 1000.0) - self.t0  # seconds

        # get all the prediction results
        predicted_human_actions = action_prediction_port.read(False)
        if predicted_human_actions is not None:
            predicted_human_actions_data = []
            for i in range(predicted_human_actions.size()):
                predicted_human_actions_data.append(predicted_human_actions.get(i))

            predicted_actions_reshaped = np.reshape(predicted_human_actions_data, (-1, number_categories))

            if len(predicted_actions_reshaped) != self.prediction_horizon:
                print(
                    "prediction values size {} and prediction horizon size {} are not equal".format(
                        len(predicted_actions_reshaped), self.prediction_horizon
                    )
                )
                return (self.p1,)

            new_time_prediction = [(time_now + i * self.time_step) for i in range(self.prediction_horizon)]
            self.t_prediction = append(self.t_prediction, new_time_prediction)
            self.action_predictions0 = append(self.action_predictions0, predicted_actions_reshaped[:, 0])
            self.action_predictions1 = append(self.action_predictions1, predicted_actions_reshaped[:, 1])
            self.action_predictions2 = append(self.action_predictions2, predicted_actions_reshaped[:, 2])
            self.action_predictions3 = append(self.action_predictions3, predicted_actions_reshaped[:, 3])

            # handle data to feed to plots
            self.t = append(self.t, time_now)
            self.prediction_now0 = append(self.prediction_now0, predicted_actions_reshaped[0, 0])
            self.prediction_now1 = append(self.prediction_now1, predicted_actions_reshaped[0, 1])
            self.prediction_now2 = append(self.prediction_now2, predicted_actions_reshaped[0, 2])
            self.prediction_now3 = append(self.prediction_now3, predicted_actions_reshaped[0, 3])

        self.p1.set_data(self.t, self.prediction_now0)
        self.p2.set_data(self.t_prediction, self.action_predictions0)

        self.p3.set_data(self.t, self.prediction_now1)
        self.p4.set_data(self.t_prediction, self.action_predictions1)

        self.p5.set_data(self.t, self.prediction_now2)
        self.p6.set_data(self.t_prediction, self.action_predictions2)

        self.p7.set_data(self.t, self.prediction_now3)
        self.p8.set_data(self.t_prediction, self.action_predictions3)

        if time_now >= self.xmax - self.plot_front_time:
            self.p1.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)
            self.p2.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)

            self.p3.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)
            self.p4.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)

            self.p5.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)
            self.p6.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)

            self.p7.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)
            self.p8.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)

            # pop the data to have faster visualization iff new data arrives
            if (predicted_human_actions is not None) and (time_now - self.t[0] > self.xmax):
                self.t_prediction = np.delete(self.t_prediction, slice(0, self.prediction_horizon))
                self.action_predictions0 = self.action_predictions0[self.prediction_horizon :]
                self.action_predictions1 = self.action_predictions1[self.prediction_horizon :]
                self.action_predictions2 = self.action_predictions2[self.prediction_horizon :]
                self.action_predictions3 = self.action_predictions3[self.prediction_horizon :]

                self.t = self.t[1:]
                self.prediction_now0 = self.prediction_now0[1:]
                self.prediction_now1 = self.prediction_now1[1:]
                self.prediction_now2 = self.prediction_now2[1:]
                self.prediction_now3 = self.prediction_now3[1:]

        return (
            self.p1,
            self.p2,
            self.p3,
            self.p4,
            self.p5,
            self.p6,
            self.p7,
            self.p8,
        )

    # Init only required for blitting to give a clean slate.
    def init(self):
        self.p1.set_data([], [])
        return (self.p1,)


plot_object = PlotInferenceResults()

ani = animation.FuncAnimation(
    plot_object.f0, plot_object.animate, interval=0, blit=True, repeat=False, cache_frame_data=False, save_count=0
)
plt.show()
