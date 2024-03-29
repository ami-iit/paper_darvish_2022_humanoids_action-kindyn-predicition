#  Copyright (C) 2023 - All rights reserved.
#  Author Contact Info: Kourosh.Darvish@gmail.com
#  SPDX-License-Identifier: BSD-3-Clause
# 

set(EXE_TARGET_NAME humanDataPreparationModule)

include(AddInstallRPATHSupport)
add_install_rpath_support(BIN_DIRS "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
  LIB_DIRS "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
  INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}"
  DEPENDS ENABLE_RPATH
  USE_LINK_PATH)

# Find required package
find_package(YARP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(HumanDynamicsEstimation)
find_package(iDynTree REQUIRED)
find_package(Threads REQUIRED)

include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include(FindPackageHandleStandardArgs)

# set cpp files
set(${EXE_TARGET_NAME}_SRC
  src/main.cpp
  src/module.cpp)

# set hpp files
set(${EXE_TARGET_NAME}_HDR
  include/module.hpp)

# add include directories to the build.
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# add an executable to the project using the specified source files.
add_executable(${EXE_TARGET_NAME} ${${EXE_TARGET_NAME}_SRC} ${${EXE_TARGET_NAME}_HDR} )

target_link_libraries(${EXE_TARGET_NAME} LINK_PUBLIC
  ${YARP_LIBRARIES}
  HumanDynamicsEstimation::HumanStateMsg
  Threads::Threads
  )

install(TARGETS ${EXE_TARGET_NAME} DESTINATION bin)

add_subdirectory(app)
