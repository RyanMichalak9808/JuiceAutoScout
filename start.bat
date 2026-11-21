@echo off
rem AutoScout one-click start (Windows): double-click start.bat
rem Finds Python 3.10+, creates a private environment (.venv) the first time,
rem then launcher.py updates AutoScout, installs packages and opens the dashboard.
rem Options are passed on: start.bat --no-update --no-browser --port 8766
rem
rem launcher.py may update this file (git pull). It runs on the last line,
rem together with exit, so nothing after it is read from the changed file.
setlocal
cd /d "%~dp0"

rem (re)create .venv when it is missing or its Python no longer runs
rem (e.g. after Python was upgraded or removed)
if exist ".venv\Scripts\python.exe" ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1 && goto run

set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=python"
if not defined PY (
    echo [AutoScout] Python 3.10 or newer is needed.
    echo             Install it with:  winget install Python.Python.3.12
    echo             or from https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^),
    echo             then close this window and double-click start.bat again.
    pause
    exit /b 1
)
if exist ".venv" echo [AutoScout] The Python environment in .venv is broken - Python was upgraded or removed? - rebuilding it...
if not exist ".venv" echo [AutoScout] First start: creating the Python environment in .venv - one time...
%PY% -m venv --clear .venv
if not exist ".venv\Scripts\python.exe" (
    echo [AutoScout] Could not create .venv - see the messages above.
    if exist ".venv" rmdir /s /q ".venv"
    pause
    exit /b 1
)

:run
".venv\Scripts\python.exe" launcher.py %* || (echo [AutoScout] Stopped with an error - see the messages above. If it keeps happening after a Python update, delete the .venv folder and start again. & pause) & exit /b
