# Copyright (C) 2023 - All rights reserved.
# Author Contact Info: Kourosh.Darvish@gmail.com
# SPDX-License-Identifier: BSD-3-Clause
#

cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 14)

set(PACKAGE_VERSION 1.0.0)
project(HumanActionIntetionPrediction
  VERSION ${PACKAGE_VERSION})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_subdirectory(src)
