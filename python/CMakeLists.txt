cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
project(smooth VERSION 1.0.0)

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
elseif(BLAS_FOUND)
   # On Windows with vcpkg, OpenBLAS may not include LAPACK routines
   # Try to find lapack library explicitly
   if(WIN32)
      find_library(LAPACK_LIB NAMES lapack)
      if(LAPACK_LIB)
         set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIB})
         message(STATUS "Found LAPACK library: ${LAPACK_LIB}")
      else()
         set(lapackblas_libraries ${BLAS_LIBRARIES})
         message(WARNING "LAPACK not found on Windows. Build may fail for modules requiring LAPACK.")
      endif()
   else()
      # Some Unix systems have OpenBLAS that includes LAPACK routines
      set(lapackblas_libraries ${BLAS_LIBRARIES})
   endif()
else()
   message(WARNING "BLAS/LAPACK not found. Build may fail for modules requiring BLAS.")
   set(lapackblas_libraries "")
endif()

# Armadillo (use correct CMake package name)
find_package(Armadillo QUIET)
if(Armadillo_FOUND)
  message(STATUS "Found Armadillo")
  include_directories(${ARMADILLO_INCLUDE_DIRS})
else()
  # carma will provide Armadillo headers via FetchContent
  message(STATUS "Armadillo not found - using carma's bundled Armadillo headers")
endif()

# Adam Core - Updated to use new core implementation
pybind11_add_module(_adamCore ../src/python/adamPython.cpp)

# Define PYTHON_BUILD for conditional compilation
target_compile_definitions(_adamCore PRIVATE PYTHON_BUILD)

# CRITICAL: Point to src/ directory so headers can be found
target_include_directories(_adamCore PRIVATE ../src)

target_link_libraries(_adamCore PRIVATE carma::carma)

# Link BLAS/LAPACK if found
if(lapackblas_libraries)
  target_link_libraries(_adamCore PRIVATE ${lapackblas_libraries})
endif()

if(Armadillo_FOUND)
  if(TARGET Armadillo::armadillo)
    target_link_libraries(_adamCore PRIVATE Armadillo::armadillo)
  else()
    target_link_libraries(_adamCore PRIVATE ${ARMADILLO_LIBRARIES})
  endif()
endif()

# Install to match module name
install(TARGETS _adamCore DESTINATION smooth/adam_general)

# LOWESS smoother module
pybind11_add_module(_lowess ../src/python/lowess.cpp)
target_compile_definitions(_lowess PRIVATE PYTHON_BUILD)
target_include_directories(_lowess PRIVATE ../src)
target_link_libraries(_lowess PRIVATE carma::carma)
if(lapackblas_libraries)
  target_link_libraries(_lowess PRIVATE ${lapackblas_libraries})
endif()
if(Armadillo_FOUND)
  if(TARGET Armadillo::armadillo)
    target_link_libraries(_lowess PRIVATE Armadillo::armadillo)
  else()
    target_link_libraries(_lowess PRIVATE ${ARMADILLO_LIBRARIES})
  endif()
endif()
install(TARGETS _lowess DESTINATION smooth/adam_general)
