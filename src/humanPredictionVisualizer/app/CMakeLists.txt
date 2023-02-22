#  Copyright (C) 2023 - All rights reserved.
#  Author Contact Info: Kourosh.Darvish@gmail.com
#  SPDX-License-Identifier: BSD-3-Clause
# 

file(GLOB scripts *.ini)
#yarp_install(FILES ${scripts} DESTINATION ${YARP_ROBOTS_INSTALL_DIR})
yarp_install(FILES ${scripts} DESTINATION ${CMAKE_INSTALL_PREFIX}/share/yarp/)
