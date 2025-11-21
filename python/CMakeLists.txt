cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

project(smooth VERSION 0.0.1)

if(SKBUILD)
  message(STATUS "The project is built using scikit-build")
endif()

# Pybind11 (fetch to avoid system dependency)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# CARMA
ADD_SUBDIRECTORY(../src/libs/carma carma)

# BLAS and LAPACK. Needed by Armadillo
find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
   set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else()
   # IS: This is where they are on my system. This might change from one OS to another
   set(lapackblas_libraries "/usr/lib/x86_64-linux-gnu/")
endif()

# Armadillo (use correct CMake package name)
find_package(Armadillo QUIET)
if(Armadillo_FOUND)
  message(STATUS "Found Armadillo")
  include_directories(${ARMADILLO_INCLUDE_DIRS})
else()
  message(WARNING "Armadillo not found. Install system package (e.g., libarmadillo-dev) or set Armadillo_DIR. Using fallback include path /usr/include.")
  set(ARMADILLO_INCLUDE_DIRS "/usr/include")
  include_directories(${ARMADILLO_INCLUDE_DIRS})
endif()

# Adam General
pybind11_add_module(_adam_general ../src/python_examples/adamGeneral.cpp)
target_include_directories(_adam_general PRIVATE ../src/python_examples/.)
target_link_libraries(_adam_general PRIVATE carma::carma)
if(Armadillo_FOUND)
  if(TARGET Armadillo::armadillo)
    target_link_libraries(_adam_general PRIVATE Armadillo::armadillo)
  else()
    target_link_libraries(_adam_general PRIVATE ${ARMADILLO_LIBRARIES})
  endif()
else()
  target_link_libraries(_adam_general PRIVATE ${lapackblas_libraries})
endif()
install(TARGETS _adam_general DESTINATION smooth/adam_general)

# Install Python source files
# install(DIRECTORY smooth DESTINATION .)

# Old experimental stuff by Leo
# code to add the pybind11 cpp module, look at demo project 
# (https://github.com/ltsaprounis/python-cpp-experiments/tree/main) for details.
#pybind11_add_module(_my_linalg ../src/python_examples/my_linalg.cpp)

# add CARMA as a subdirectory
#add_subdirectory(../src/libs/carma build)
#target_link_libraries(_my_linalg
#    PRIVATE carma::carma
#    ${ARMADILLO_LIBRARIES}
#)

# install(TARGETS _my_linalg DESTINATION smooth/my_linalg)

