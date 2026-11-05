@echo off
setlocal
cd /d "%~dp0"
echo === Prawn live detection: one-time install ===
echo.

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    python --version >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Python was not found on this computer.
    echo.
    echo 1. Install Python 3.10 or newer from https://www.python.org/downloads/
    echo 2. In the installer, tick "Add python.exe to PATH".
    echo 3. Then double-click install.bat again.
    echo.
    pause
    exit /b 1
)

%PY% --version
%PY% -m venv .venv || goto :fail
.venv\Scripts\python.exe -m pip install --upgrade pip || goto :fail
.venv\Scripts\python.exe -m pip install ultralytics pygrabber || goto :fail

echo.
echo Install finished. Now double-click run.bat
pause
exit /b 0

:fail
echo.
echo Install FAILED. Please send a screenshot of this window to the person who sent you this package.
pause
exit /b 1
