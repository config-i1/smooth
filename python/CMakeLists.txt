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
   # Fallback: explicitly specify the libraries
   set(lapackblas_libraries "/usr/lib/x86_64-linux-gnu/libblas.so" "/usr/lib/x86_64-linux-gnu/liblapack.so")
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

# Adam Core - Updated to use new core implementation
pybind11_add_module(_adamCore ../src/python/adamPython.cpp)

# CRITICAL: Point to src/ directory so headers can be found
target_include_directories(_adamCore PRIVATE ../src)

target_link_libraries(_adamCore PRIVATE carma::carma)

# Always link BLAS/LAPACK
target_link_libraries(_adamCore PRIVATE ${lapackblas_libraries})

if(Armadillo_FOUND)
  if(TARGET Armadillo::armadillo)
    target_link_libraries(_adamCore PRIVATE Armadillo::armadillo)
  else()
    target_link_libraries(_adamCore PRIVATE ${ARMADILLO_LIBRARIES})
  endif()
endif()

# Install to match module name
install(TARGETS _adamCore DESTINATION smooth)
