#!/bin/sh
# Copyright (C) 2023 - All rights reserved.
# Author Contact Info: Kourosh.Darvish@gmail.com
# SPDX-License-Identifier: BSD-3-Clause
#

python scripts/animate_action_prediction.py &
python scripts/animate_joint_angle_prediction.py &
python scripts/animate_wrench_prediction.py &
humanPredictionVisualizerModule --from HumanPredictionVisualizer.ini
