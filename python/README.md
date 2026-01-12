# PY-SMOOTH
Python version of smooth.

**⚠️WORK IN PROGRESS⚠️**

# Notes:

1. To build the project using [scikit-build-core](https://github.com/scikit-build/scikit-build-core) simply do `pip install -e .`
2. The files in python/smooth/mylinalg and src/my_linalg.cpp are example files for carma and pybind11 and will be removed later.
3. The latest CARMA releases support numpy 2. We will need to update this at some point soon, for more info check
the [CARMA requirements](https://github.com/RUrlus/carma#requirements)

# TODOs:
- [X] use scikit-build-core to compile the C++ modules
- [X] (TBC) Refactor src code to use armadillo instead of rcpparmadillo in common Cpp code.
- [ ] CI pipelines for tests
- [ ] Add sphinx docs for python package
- [ ] Check how to automate the Armadillo installation either as a github submodule or using [cmake's FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html#fetchcontent) like in [carma](https://github.com/RUrlus/carma/blob/2fbc2e6faf2e40e41003c06cbb13744405732b5f/integration_test/CMakeLists.txt#L36)
