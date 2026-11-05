@echo off
setlocal
cd /d "%~dp0"

rem ===================== Settings =====================
rem Camera number: 0 = first camera. If the wrong camera opens (or none), try 1, then 2.
set CAMERA=0
rem Requested camera resolution (the camera uses the closest size it supports).
set WIDTH=1920
set HEIGHT=1080
rem Also save the clean camera video (yes/no). Needed for training; about 1 GB per hour.
set SAVE_RAW=yes
set MODEL=prawn_yolo11n_pose_v0.1.pt
rem ====================================================

if not exist .venv\Scripts\python.exe (
    echo Please double-click install.bat first.
    pause
    exit /b 1
)

set "EXTRA="
if /i "%SAVE_RAW%"=="no" set "EXTRA=--no-raw"

echo Starting the camera... press q in the video window to stop.
.venv\Scripts\python.exe live_prawn_detect.py --source %CAMERA% --width %WIDTH% --height %HEIGHT% --model %MODEL% --kpts --record runs %EXTRA%

echo.
echo Done. The results are in the "runs" folder (the newest folder is this run).
echo Please zip that folder and send it back.
if exist runs start "" runs
pause
