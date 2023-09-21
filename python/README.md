# PY-SMOOTH
Python version of smooth.

**⚠️WORK IN PROGRESS⚠️**

# Notes:

To build the project using [scikit-build-core](https://github.com/scikit-build/scikit-build-core) simply do `pip install -e .`

# TODOs:
- [X] use scikit-build-core to compile the C++ modules
- [ ] CI pipelines for tests
- [ ] Add sphinx docs for python package
- [ ] Check how to automate the Armadillo installation either as a github submodule or using [cmake's FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html#fetchcontent) like in [carma](https://github.com/RUrlus/carma/blob/2fbc2e6faf2e40e41003c06cbb13744405732b5f/integration_test/CMakeLists.txt#L36)
