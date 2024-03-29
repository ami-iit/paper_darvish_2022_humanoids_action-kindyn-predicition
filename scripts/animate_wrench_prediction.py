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
import numpy as np
import yarp
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot

from kindyn_prediction.utils.dataset_utils import current_milli_time

# yarp related code
if not yarp.Network.checkNetwork():
    print("[main] Unable to find YARP network")
yarp.Network.init()
rf = yarp.ResourceFinder()
rf.setDefaultContext("myContext")
rf.setDefaultConfigFile("default.ini")

human_kin_dyn_port = yarp.BufferedPortBottle()
human_kin_dyn_port.open("/test_visualization/humanDynamics:i")
motion_prediction_port = yarp.BufferedPortVector()
motion_prediction_port.open("/test_visualization/dynamicsPredictionAll:i")

is_connected_human_kindyn = yarp.Network.connect(
    "/humanDataAcquisition/humanKinDyn:o", "/test_visualization/humanDynamics:i"
)
is_connected_motion_prediction = yarp.Network.connect(
    "/test_moe/dynamicPredictionAll:o", "/test_visualization/dynamicsPredictionAll:i"
)
print(f"human kindyn port is connected: {is_connected_human_kindyn}")
print(f"motion prediction port is connected: {is_connected_motion_prediction}")
yarp.delay(0.5)


class PlotInferenceResults:
    def __init__(self):
        # related to figure
        font = {"size": 15}
        matplotlib.rc("font", **font)
        self.variable_idx_prediction = 2
        self.variable_idx_ground_truth = 2 * 66 + 2
        self.variableName = "l_fz"
        self.xmin = 0.0
        self.xmax = 6.5
        self.plot_front_time = 1.2
        self.f0 = figure(num=0, figsize=(8, 3.5))  # , dpi = 100)
        self.ax01 = self.f0.subplots()  # 2grid((1, 1), (0, 0))
        self.ax01.set_ylim(-100, 1000)
        self.ax01.set_xlim(self.xmin, self.xmax)
        self.t = np.zeros(0)
        self.t0 = current_milli_time() / 1000.0  # seconds
        self.joint_values = np.zeros(0)
        (self.p1,) = self.ax01.plot(self.t, self.joint_values, "b-", linewidth=5)
        self.t_prediction = np.zeros(0)
        self.joint_predictions = np.zeros(0)
        (self.p2,) = self.ax01.plot(self.t_prediction, self.joint_predictions, "o", color="k", markersize=4, alpha=0.1)

        self.ax01.grid(True)

        self.x = 0.0

        # related to the data
        self.timer = current_milli_time()
        self.counter = 0
        self.time_length = 100
        self.human_kin_dyn_data = []

        self.prediction_horizon = 25
        self.time_step = 0.04
        self.output_size = 12

        return

    def animate(self, dummy):
        # read human current data:
        # data manipulation
        print(f"timer: {current_milli_time() - self.timer}")
        self.timer = current_milli_time()
        time_now = (current_milli_time() / 1000.0) - self.t0  # seconds

        # set the human current joint values
        human_kin_dyn = human_kin_dyn_port.read(False)
        if human_kin_dyn is not None:
            tmp_joint = human_kin_dyn.get(self.variable_idx_ground_truth).asFloat64()
        else:
            return (
                self.p1,
                self.p2,
            )

        # get all the prediction results
        human_kin_dyn_prediction = motion_prediction_port.read(False)
        if human_kin_dyn_prediction is not None:
            human_kin_dyn_prediction_data = []
            for i in range(self.variable_idx_prediction, human_kin_dyn_prediction.size(), self.output_size):
                human_kin_dyn_prediction_data.append(human_kin_dyn_prediction.get(i))

            if len(human_kin_dyn_prediction_data) != self.prediction_horizon:
                print(
                    "prediction values size {} and prediction horizon size {} are not equal".format(
                        len(human_kin_dyn_prediction_data), self.prediction_horizon
                    )
                )
                return (self.p1,)

            new_time_prediction = [(time_now + i * self.time_step) for i in range(self.prediction_horizon)]
            self.t_prediction = append(self.t_prediction, new_time_prediction)
            self.joint_predictions = append(self.joint_predictions, human_kin_dyn_prediction_data)

        # handle data to feed to plots
        self.joint_values = append(self.joint_values, tmp_joint)
        self.t = append(self.t, time_now)

        self.x += 0.05
        # handling figure
        self.p2.set_data(self.t_prediction, self.joint_predictions)
        self.p1.set_data(self.t, self.joint_values)

        if time_now >= self.xmax - self.plot_front_time:
            self.p1.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)
            self.p2.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)

            if human_kin_dyn_prediction is not None and (time_now - self.t[0] > self.xmax):
                self.t_prediction = self.t_prediction[self.prediction_horizon :]
                self.joint_predictions = self.joint_predictions[self.prediction_horizon :]

            if human_kin_dyn is not None:
                self.t = self.t[1:]
                self.joint_values = self.joint_values[1:]

        return (
            self.p1,
            self.p2,
        )

    # Init only required for blitting to give a clean slate.
    def init(self):
        self.p1.set_data([], [])
        return (self.p1,)


plot_object = PlotInferenceResults()

ani = animation.FuncAnimation(plot_object.f0, plot_object.animate, interval=20, blit=False, repeat=False)
plt.show()
