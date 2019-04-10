cmake_minimum_required(VERSION 3.12)
project(karma C)

include(ExternalProject)

set(CMAKE_C_STANDARD 11)

# Set LAPACK configuration. Example using openblas with 64 bit indexing:
set(LAPACK openblas)
set(LAPACK_INCLUDE /usr/include)
set(LAPACK_LIB /usr/lib)
set(LAPACK_FLAGS "-DHAVE_LAPACK_CONFIG_H -DLAPACK_ILP64")

# Enable openmp.
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

# Enable optimization and use LAPACK flags.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 ${LAPACK_FLAGS}")

# Set install directory for external libraries.
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

# Add chronos library to project.
ExternalProject_Add(chronos_lib GIT_REPOSITORY https://github.com/pierfied/chronos
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION})

# Add include and lib directories for chronos and LAPACK.
include_directories(${EXTERNAL_INSTALL_LOCATION}/include ${LAPACK_INCLUDE})
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib ${LAPACK_LIB})

# Create the shared library for karma and the SVD executable.
add_library(karma SHARED src/karma.c src/karma.h src/svd.c src/svd.h)
add_executable(karma_svd src/svd.c src/svd.h)

# Set dependencies so that chronos is built before linking to karma.
add_dependencies(karma chronos_lib)

# Link karma to the required libraries.
target_link_libraries(karma m libchronos.a ${LAPACK})
target_link_libraries(karma_svd ${LAPACK})