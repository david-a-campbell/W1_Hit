@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "DIR=%%~fI"
set "VENV=%DIR%\.venv"
set "REQ=%SCRIPT_DIR%requirements.txt"
set "CONSTRAINTS=%SCRIPT_DIR%.w1_hit_constraints.txt"
set "PYTHON_BIN="
set "PYTHON_EXE="
set "PYTHON_VERSION="

set "CONDA_PREFIX="
set "CONDA_DEFAULT_ENV="
set "CONDA_EXE="
set "CONDA_PYTHON_EXE="
set "_CE_CONDA="
set "_CE_M="
set "PYTHONHOME="
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

echo W1 Hit installer
echo Device dir: %DIR%
echo.

if not exist "%REQ%" (
  echo ERROR: requirements.txt not found at: %REQ%
  echo Make sure this installer is in the same folder as requirements.txt.
  echo.
  pause
  exit /b 1
)

call :FindPython
if errorlevel 1 (
  call :InstallPython
  if errorlevel 1 exit /b 1
  call :FindPython
  if errorlevel 1 (
    echo ERROR: Python was installed, but this installer could not find it.
    echo Close this window, open a new Command Prompt, and run the installer again.
    echo.
    pause
    exit /b 1
  )
)

%PYTHON_BIN% -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12 else 1)"
if errorlevel 1 (
  echo ERROR: Selected Python is not 3.10, 3.11, or 3.12.
  echo.
  pause
  exit /b 1
)

for /f "delims=" %%P in ('%PYTHON_BIN% -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%P"
for /f "delims=" %%P in ('%PYTHON_BIN% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do set "PYTHON_VERSION=%%P"

echo Using Python: %PYTHON_EXE%
echo Python version: %PYTHON_VERSION%
echo.

if exist "%VENV%\Scripts\python.exe" (
  set "VENV_VERSION="
  set "VENV_EXE="
  for /f "delims=" %%P in ('"%VENV%\Scripts\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "VENV_VERSION=%%P"
  for /f "delims=" %%P in ('"%VENV%\Scripts\python.exe" -c "import sys; print(sys.executable.lower())" 2^>nul') do set "VENV_EXE=%%P"
  echo !VENV_EXE! | findstr /i "conda anaconda miniconda mambaforge miniforge" >nul
  if not errorlevel 1 set "VENV_VERSION=blocked"
  if not "!VENV_VERSION!"=="3.10" if not "!VENV_VERSION!"=="3.11" if not "!VENV_VERSION!"=="3.12" (
    echo Existing virtual environment is incompatible. Recreating it...
    rmdir /s /q "%VENV%"
  )
)

if not exist "%VENV%" (
  echo Creating virtual environment...
  %PYTHON_BIN% -m venv "%VENV%"
  if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    echo.
    pause
    exit /b 1
  )
)

> "%CONSTRAINTS%" echo numpy^<2

echo Upgrading pip tools...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto PipError

echo Installing requirements...
"%VENV%\Scripts\python.exe" -m pip install --force-reinstall "numpy<2"
if errorlevel 1 goto PipError
"%VENV%\Scripts\python.exe" -m pip install -c "%CONSTRAINTS%" -r "%REQ%"
if errorlevel 1 goto PipError
"%VENV%\Scripts\python.exe" -m pip install --force-reinstall "numpy<2"
if errorlevel 1 goto PipError

echo.
echo Verifying install...
"%VENV%\Scripts\python.exe" -c "import importlib, sys; mods=['numpy','torch','mido']; [print(m, getattr(importlib.import_module(m), '__version__', 'installed')) for m in mods]"
if errorlevel 1 (
  echo ERROR: Install verification failed.
  echo.
  pause
  exit /b 1
)

echo.
echo Done. You can now run the device.
pause
exit /b 0

:PipError
echo.
echo ERROR: Python package installation failed.
echo Check your internet connection, then run this installer again.
echo.
pause
exit /b 1

:FindPython
set "PYTHON_BIN="

for %%V in (3.11 3.10 3.12) do (
  if not defined PYTHON_BIN (
    py -%%V -c "import sys; print(sys.executable)" >nul 2>nul
    if !errorlevel! equ 0 (
      for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable.lower())"') do set "CHECK_EXE=%%P"
      echo !CHECK_EXE! | findstr /i "conda anaconda miniconda mambaforge miniforge" >nul
      if errorlevel 1 set "PYTHON_BIN=py -%%V"
    )
  )
)

if defined PYTHON_BIN exit /b 0

for %%C in (python3.11 python3.10 python3.12 python) do (
  if not defined PYTHON_BIN (
    where %%C >nul 2>nul
    if !errorlevel! equ 0 (
      %%C -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12 else 1)" >nul 2>nul
      if !errorlevel! equ 0 (
        for /f "delims=" %%P in ('%%C -c "import sys; print(sys.executable.lower())"') do set "CHECK_EXE=%%P"
        echo !CHECK_EXE! | findstr /i "conda anaconda miniconda mambaforge miniforge windowsapps" >nul
        if errorlevel 1 set "PYTHON_BIN=%%C"
      )
    )
  )
)

if defined PYTHON_BIN exit /b 0
exit /b 1

:InstallPython
echo No compatible Python found. Installing Python 3.11...
where winget >nul 2>nul
if errorlevel 1 (
  echo ERROR: Could not find winget.
  echo Install Python 3.11 from python.org, check "Add python.exe to PATH", then run this installer again.
  echo.
  pause
  exit /b 1
)

winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
  echo ERROR: Python 3.11 installation failed.
  echo Install Python 3.11 from python.org, check "Add python.exe to PATH", then run this installer again.
  echo.
  pause
  exit /b 1
)
exit /b 0
