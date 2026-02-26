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
  GIT_TAG v2.12.0)
FetchContent_MakeAvailable(pybind11)

# CARMA
ADD_SUBDIRECTORY(../src/libs/carma carma)

# ==============================================================================
# BLAS/LAPACK Detection
# ==============================================================================

if(WIN32)
  message(STATUS "Windows build: disabling BLAS/LAPACK in Armadillo to avoid unresolved symbols")
  # Disable use of BLAS/LAPACK in Armadillo on Windows
  add_compile_definitions(ARMA_DONT_USE_BLAS ARMA_DONT_USE_LAPACK)
  set(BLAS_FOUND FALSE)
  set(LAPACK_FOUND FALSE)
  set(lapackblas_libraries "")
else()
  # Standard BLAS and LAPACK search for Linux/macOS
  find_package(BLAS)
  find_package(LAPACK)

  if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
    message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
  elseif(BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES})
  else()
    message(WARNING "BLAS/LAPACK not found. Build may fail for modules requiring BLAS.")
    set(lapackblas_libraries "")
  endif()
endif()

# ==============================================================================
# End BLAS/LAPACK Detection
# ==============================================================================

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

# Link BLAS/LAPACK if found (not on Windows)
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

# EigenCalc module for eigenvalue calculations
pybind11_add_module(_eigenCalc ../src/python/eigenCalc.cpp)
target_compile_definitions(_eigenCalc PRIVATE PYTHON_BUILD)
target_include_directories(_eigenCalc PRIVATE ../src)
target_link_libraries(_eigenCalc PRIVATE carma::carma)
if(lapackblas_libraries)
  target_link_libraries(_eigenCalc PRIVATE ${lapackblas_libraries})
endif()
if(Armadillo_FOUND)
  if(TARGET Armadillo::armadillo)
    target_link_libraries(_eigenCalc PRIVATE Armadillo::armadillo)
  else()
    target_link_libraries(_eigenCalc PRIVATE ${ARMADILLO_LIBRARIES})
  endif()
endif()
install(TARGETS _eigenCalc DESTINATION smooth/adam_general)
