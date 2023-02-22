# Copyright (C) 2020 Fondazione Istituto Italiano di Tecnologia (IIT)
# All Rights Reserved.
# Authors: Kourosh Darvish <kourosh.darvish@iit.it>

file(GLOB scripts *.ini)
yarp_install(FILES ${scripts} DESTINATION ${CMAKE_INSTALL_PREFIX}/share/yarp/)
