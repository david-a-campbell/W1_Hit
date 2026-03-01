@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "DIR=%SCRIPT_DIR%.."
set "VENV=%DIR%\.venv"

echo W1 Hit installer
echo Device dir: %DIR%

where python >nul 2>nul
if %errorlevel% neq 0 (
  echo ERROR: python not found. Install Python 3 first.
  exit /b 1
)

if not exist "%VENV%" (
  python -m venv "%VENV%"
)

"%VENV%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
"%VENV%\Scripts\pip.exe" install -r "%SCRIPT_DIR%requirements.txt"

echo Done.
endlocal