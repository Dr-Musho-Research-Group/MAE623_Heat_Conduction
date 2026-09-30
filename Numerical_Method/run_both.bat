@echo off
setlocal
rem Double-click this file to run both examples in separate windows.
rem Each command window stays open so completion messages/errors are visible.
rem Change to this batch file's folder, even when launched from elsewhere.
cd /d "%~dp0"

rem Prefer the user's Python 3.10 installation, then try Python on PATH.
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if exist "%PYTHON_EXE%" goto run_examples
set "PYTHON_EXE=python"
where python >nul 2>nul
if not errorlevel 1 goto run_examples
set "PYTHON_EXE=py"
where py >nul 2>nul
if not errorlevel 1 goto run_examples

echo Python was not found. Install Python 3 and add it to PATH.
pause
exit /b 1

:run_examples
rem Install dependencies once if needed: pip install numpy matplotlib
rem Each script writes a contour PNG without opening a figure window.
rem Close the command windows when finished.
start "Explicit FTCS" "%ComSpec%" /k ""%PYTHON_EXE%" "%~dp0transient_1d_explicit.py""
start "Implicit Backward Euler" "%ComSpec%" /k ""%PYTHON_EXE%" "%~dp0transient_1d_implicit.py""
endlocal
