#  Copyright (C) 2023 - All rights reserved.
#  Author Contact Info: Kourosh.Darvish@gmail.com
#  SPDX-License-Identifier: BSD-3-Clause
# 

set(EXE_TARGET_NAME humanPredictionVisualizerModule)

option(ENABLE_RPATH "Enable RPATH for this library" ON)
mark_as_advanced(ENABLE_RPATH)
include(AddInstallRPATHSupport)
add_install_rpath_support(BIN_DIRS "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
  LIB_DIRS "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
  INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}"
  DEPENDS ENABLE_RPATH
  USE_LINK_PATH)

# Find required package
find_package(YARP REQUIRED)
find_package(iDynTree REQUIRED)

# set cpp files
set(${EXE_TARGET_NAME}_SRC
  src/main.cpp)

# add include directories to the build.
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# add an executable to the project using the specified source files.
add_executable(${EXE_TARGET_NAME} ${${EXE_TARGET_NAME}_SRC} )

target_link_libraries(${EXE_TARGET_NAME} LINK_PUBLIC
  YARP::YARP_OS
  YARP::YARP_dev
  YARP::YARP_init
  iDynTree::idyntree-core
  iDynTree::idyntree-modelio-urdf
  iDynTree::idyntree-visualization)

install(TARGETS ${EXE_TARGET_NAME} DESTINATION bin)

add_subdirectory(app)
