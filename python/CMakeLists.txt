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

# BLAS / LAPACK are required by Armadillo's ``eig_gen`` (used by
# ``ar_polynomial_bounds`` for ARIMA reapply and ``eigen_bounds`` for
# admissible bounds). Without LAPACK linked, ``eig_gen`` throws at
# runtime, so we must detect them robustly on every platform.
#
# Detection strategy:
#   1. Try CONFIG-mode ``find_package(OpenBLAS CONFIG)`` first — vcpkg
#      ships ``OpenBLASConfig.cmake`` for its ``openblas`` port and
#      that port is a full LAPACK build (BUILD_WITHOUT_LAPACK=OFF by
#      default), so a single CONFIG package resolves both BLAS and
#      LAPACK on Windows wheels. MODULE-mode FindBLAS/FindLAPACK never
#      reliably picked up the vcpkg install layout, even with
#      BLA_VENDOR=OpenBLAS, hence the silent fall-through to
#      ARMA_DONT_USE_LAPACK and the ``eig_gen()`` runtime crash.
#   2. Fall back to MODULE-mode ``find_package(BLAS)`` /
#      ``find_package(LAPACK)`` for Linux / macOS, where the system
#      packages (openblas-devel + lapack-devel on Linux,
#      brew's openblas on macOS) are found by the canonical search.
#   3. Final fallback only when the dependency is genuinely absent.
find_package(OpenBLAS CONFIG QUIET)
if(OpenBLAS_FOUND)
  message(STATUS "Found OpenBLAS via CONFIG (vcpkg-style); "
                 "provides BLAS + LAPACK in a single library.")
  set(BLAS_FOUND TRUE)
  set(LAPACK_FOUND TRUE)
  set(lapackblas_libraries OpenBLAS::OpenBLAS)
else()
  find_package(BLAS)
  find_package(LAPACK)
  if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
    message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
  elseif(BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES})
    add_compile_definitions(ARMA_DONT_USE_LAPACK)
    message(WARNING "LAPACK not found; eig_gen / ar_polynomial_bounds disabled.")
  else()
    message(WARNING "BLAS/LAPACK not found. Disabling BLAS/LAPACK in Armadillo. "
                    "Modules requiring eig_gen / eigvalsh will fail at runtime.")
    add_compile_definitions(ARMA_DONT_USE_BLAS ARMA_DONT_USE_LAPACK)
    set(lapackblas_libraries "")
  endif()
endif()

# Fail loudly on Windows if LAPACK still isn't found — silently
# shipping a wheel without LAPACK leaves every ARIMA / admissible
# code path broken with a runtime crash, which is worse than a
# build failure that surfaces the misconfiguration immediately.
if(WIN32 AND NOT LAPACK_FOUND)
  message(FATAL_ERROR
    "LAPACK not found on Windows. The wheel build expects vcpkg's "
    "openblas (which bundles LAPACK) — make sure "
    "``CIBW_BEFORE_ALL_WINDOWS`` runs ``vcpkg install "
    "openblas:x64-windows-release`` and that "
    "``CMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake`` "
    "is exported via ``CIBW_ENVIRONMENT_WINDOWS``.")
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

# numDeriv module for numerical derivatives (pracma-exact Hessian)
pybind11_add_module(_numDeriv ../src/python/numDeriv.cpp)
target_compile_definitions(_numDeriv PRIVATE PYTHON_BUILD)
target_include_directories(_numDeriv PRIVATE ../src)
target_link_libraries(_numDeriv PRIVATE carma::carma)
if(lapackblas_libraries)
  target_link_libraries(_numDeriv PRIVATE ${lapackblas_libraries})
endif()
if(Armadillo_FOUND)
  if(TARGET Armadillo::armadillo)
    target_link_libraries(_numDeriv PRIVATE Armadillo::armadillo)
  else()
    target_link_libraries(_numDeriv PRIVATE ${ARMADILLO_LIBRARIES})
  endif()
endif()
install(TARGETS _numDeriv DESTINATION smooth/adam_general)
