/*
* Copyright (C) 2023 - All rights reserved.
* Author Contact Info: Kourosh.Darvish@gmail.com
* SPDX-License-Identifier: BSD-3-Clause
*/


// YARP
#include <module.hpp>
#include <thread>
#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/RFModule.h>

int main(int argc, char *argv[]) {
  // initialise yarp network
  yarp::os::Network yarp;
  if (!yarp.checkNetwork()) {
    yError() << "[main] Unable to find YARP network";
    return EXIT_FAILURE;
  }

  // prepare and configure the resource finder
  yarp::os::ResourceFinder &rf =
      yarp::os::ResourceFinder::getResourceFinderSingleton();

  rf.setDefaultConfigFile("humanDataForRiskAssessment.ini");

  rf.configure(argc, argv);

  // create the module
  HumanDataAcquisitionModule module;

  if (!module.configure(rf)) {
    yError() << "[main] cannot configure the module";
    return 1;
  }
  try {

    std::thread run_thread(&HumanDataAcquisitionModule::updateModule, &module);

    std::thread keyboard_thread(&HumanDataAcquisitionModule::keyboardHandler,
                                &module);

    keyboard_thread.join();
    run_thread.join();

  } catch (std::exception &e) {
    std::cerr << "Unhandled Exception: " << e.what() << std::endl;
  }

  return 0;
}
