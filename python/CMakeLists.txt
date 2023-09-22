cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

project(smooth VERSION 0.0.1)

if(SKBUILD)
  message(STATUS "The project is built using scikit-build")
  endif()

# Pybind11
find_package(pybind11 REQUIRED)
# Armadillo
find_package(armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# code to add the pybind11 cpp module, look at demo project 
# (https://github.com/ltsaprounis/python-cpp-experiments/tree/main) for details.
pybind11_add_module(_my_linalg ../src/my_linalg.cpp)

# add CARMA as a subdirectory
add_subdirectory(../src/libs/carma build)
target_link_libraries(_my_linalg 
    PRIVATE carma::carma
    ${ARMADILLO_LIBRARIES}
)

install(TARGETS _my_linalg DESTINATION smooth/my_linalg)
