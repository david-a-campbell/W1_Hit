@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "DIR=%SCRIPT_DIR%.."
set "VENV=%DIR%\.venv"
set "PYTHON_BIN="
set "PYTHON_VERSION="

echo W1 Hit installer
echo Device dir: %DIR%
echo.

REM Prefer Python versions known to work with torch==2.2.2.
for %%V in (3.11 3.10) do (
  if not defined PYTHON_BIN (
    py -%%V -c "import sys; print(sys.executable)" >nul 2>nul
    if !errorlevel! equ 0 (
      set "PYTHON_BIN=py -%%V"
    )
  )
)

REM Fallback to python only if it is 3.10 or 3.11.
if not defined PYTHON_BIN (
  where python >nul 2>nul
  if %errorlevel% equ 0 (
    for /f "delims=" %%P in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "PYTHON_VERSION=%%P"
    if "!PYTHON_VERSION!"=="3.11" set "PYTHON_BIN=python"
    if "!PYTHON_VERSION!"=="3.10" set "PYTHON_BIN=python"
  )
)

if not defined PYTHON_BIN (
  echo ERROR: Compatible Python not found.
  echo W1 Hit needs Python 3.10 or 3.11 because PyTorch does not support every Python version.
  echo.
  echo Install Python 3.11 from https://www.python.org/downloads/ or Microsoft Store,
  echo then run this installer again.
  pause
  exit /b 1
)

for /f "delims=" %%P in ('%PYTHON_BIN% -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%P"
for /f "delims=" %%P in ('%PYTHON_BIN% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do set "PYTHON_VERSION=%%P"

echo Using Python: %PYTHON_EXE%
echo Python version: %PYTHON_VERSION%
echo.

REM Recreate venv if it exists but uses an incompatible Python.
if exist "%VENV%\Scripts\python.exe" (
  for /f "delims=" %%P in ('"%VENV%\Scripts\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "VENV_VERSION=%%P"
  if not "!VENV_VERSION!"=="3.10" if not "!VENV_VERSION!"=="3.11" (
    echo Existing venv uses Python !VENV_VERSION!, recreating it...
    rmdir /s /q "%VENV%"
  )
)

if not exist "%VENV%" (
  echo Creating virtual environment...
  %PYTHON_BIN% -m venv "%VENV%"
  if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
  )
)

echo Upgrading pip tools...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 (
  echo ERROR: Failed to upgrade pip tools.
  pause
  exit /b 1
)

REM Install NumPy first and pin below 2 for compatibility with torch 2.2.2.
echo Installing NumPy...
"%VENV%\Scripts\python.exe" -m pip install --force-reinstall "numpy<2"
if %errorlevel% neq 0 (
  echo ERROR: Failed to install NumPy.
  pause
  exit /b 1
)

echo Installing requirements...
"%VENV%\Scripts\python.exe" -m pip install -r "%SCRIPT_DIR%requirements.txt"
if %errorlevel% neq 0 (
  echo ERROR: Failed to install requirements.
  pause
  exit /b 1
)

echo.
echo Verifying install...
"%VENV%\Scripts\python.exe" -c "import numpy, torch, mido; print('numpy', numpy.__version__); print('torch', torch.__version__); print('mido', mido.__version__)"
if %errorlevel% neq 0 (
  echo ERROR: Install verification failed.
  pause
  exit /b 1
)

echo.
echo Done.
pause
endlocal
