cmake_minimum_required(VERSION 3.12)
project(karma C)

set(CMAKE_C_STANDARD 11)

include(ExternalProject)

# Set install directory for external libraries.
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

# Add chronos library to project.
ExternalProject_Add(chronos_lib GIT_REPOSITORY https://github.com/pierfied/chronos
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION})

# Add OpenBLAS library to project.
ExternalProject_Add(openblas_lib GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DINTERFACE64=1)

# Add include and lib directories for external projects.
include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

# Flags to enable 64 bit indexing for openblas.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_LAPACK_CONFIG_H  -DLAPACK_ILP64")

# Create the shared library for karma.
add_library(karma SHARED src/karma.c src/karma.h)

# Set dependencies so that chronos and openblas are built before linking to karma.
# Disable during development or clion will rebuild the dependencies every time karma is built.
add_dependencies(karma chronos_lib openblas_lib)

# Link karma to the static chronos and openblas libraries.
target_link_libraries(karma libchronos.a libopenblas.a)