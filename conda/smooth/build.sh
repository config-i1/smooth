#!/bin/bash
set -ex

cd "${SRC_DIR}/python"

# Replace FetchContent(pybind11) with find_package(pybind11)
# conda provides pybind11, and network access is not available during builds
sed -i 's/include(FetchContent)//' CMakeLists.txt
sed -i '/FetchContent_Declare/,/FetchContent_MakeAvailable(pybind11)/c\find_package(pybind11 REQUIRED)' CMakeLists.txt

${PYTHON} -m pip install . -vv --no-deps --no-build-isolation
