cmake_minimum_required(VERSION 3.12)
project(lmapr C)

set(CMAKE_C_STANDARD 11)

include(ExternalProject)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(chronos_lib GIT_REPOSITORY https://github.com/pierfied/chronos
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION})

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

add_library(lmapr SHARED src/lmapr.c src/lmapr.h)
add_dependencies(lmapr chronos_lib)
target_link_libraries(lmapr libchronos.a)