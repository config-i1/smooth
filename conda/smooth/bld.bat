@echo off
cd "%SRC_DIR%\python"

:: Replace FetchContent(pybind11) with find_package(pybind11)
:: conda provides pybind11, and network access is not available during builds
powershell -Command "(Get-Content CMakeLists.txt) -replace 'include\(FetchContent\)', '' | Set-Content CMakeLists.txt"
powershell -Command "$c = Get-Content CMakeLists.txt -Raw; $c = $c -replace '(?s)FetchContent_Declare\(.*?FetchContent_MakeAvailable\(pybind11\)', 'find_package(pybind11 REQUIRED)'; Set-Content CMakeLists.txt $c"

%PYTHON% -m pip install . -vv --no-deps --no-build-isolation
if errorlevel 1 exit /B 1
