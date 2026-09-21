@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto check_environment
py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  py -3.12 -m venv .venv
) else (
  python -c "import sys; sys.exit(sys.version_info[:2] < (3, 12))" >nul 2>nul
  if errorlevel 1 goto need_python
  python -m venv .venv
)
if errorlevel 1 goto fail
:check_environment
".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info[:2] < (3, 12))" >nul 2>nul
if errorlevel 1 goto need_python
".venv\Scripts\python.exe" -c "from pathlib import Path; from importlib.metadata import version; specs=[line.strip().split('==') for line in Path('requirements.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]; assert all(version(name) == expected for name, expected in specs)" >nul 2>nul
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" launch.py %*
if errorlevel 1 goto fail
exit /b 0
:need_python
echo.
echo Python 3.12 or later is required. Python 3.12 is the tested version.
echo Install it from https://www.python.org/downloads/ and enable the Python launcher.
echo If this folder already contains an older .venv, rename it and try again.
:fail
echo.
echo Startup failed. See README.md for manual setup with Python 3.12 or later.
pause
exit /b 1
