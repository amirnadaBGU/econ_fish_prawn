@echo off
setlocal
cd /d "%~dp0"

rem ===================== Settings =====================
rem What to detect: prawn or fish (each has its own model, see MODEL below).
set TARGET=fish
rem Fish only: improve each image before the model, the same way the fish training images
rem   were made - CLAHE (local contrast) + sharpening (yes/no). The prawn model never uses it.
rem   Only what the model sees changes; raw.mkv keeps the original camera images.
rem   Costs about 20 ms per image at 1080p (about 50 ms at 4K).
set PREPROCESS=yes
rem Camera: a name (part of it is enough) or a number.
rem   See3CAM = the e-con See3CAM_CU83 camera (found by name, whatever its number).
rem   0, 1, 2 = camera number, for any other camera.
set CAMERA=See3CAM
rem Camera mode (See3CAM_CU83). Write one of these after MODE=
rem   1080p  color 1920x1080, 30 fps.  Recommended - the model was trained on color images.
rem   4k     color 3840x2160, 15 fps.  More detail, but much slower (about 4 fps on a laptop)
rem          and very large videos - use with VIDEO_QUALITY=medium or low.
rem   720p   color 1280x720, 30 fps.   For a slow computer.
rem   ir     infrared, grey 1920x1080, 30 fps.  See "About IR" below.
rem   ir4k   infrared, grey 3840x2160, 15 fps.  Probably the same detail as ir, enlarged.
rem About IR (infrared):
rem   The camera also sees near-infrared light (invisible to the eye), so it can film in the
rem   dark - but only with an infrared lamp (illuminator) shining on the scene. It is NOT a
rem   thermal camera: it does not see heat. The standard camera has a 940 nm lens, so use a
rem   940 nm lamp (an 850 nm lens version also exists - then use an 850 nm lamp).
rem   The image is grey and its brightness is adjusted automatically in every frame.
rem   The model learned from color images, so it may find fewer prawns / fish in IR - test it first.
rem Not available here: the camera's combined color+IR modes (they need e-con's own software).
rem For another camera (CAMERA=0, 1 ...) use 1080p, 4k or 720p only.
set MODE=720p
rem Record the session into the "runs" folder (yes/no). no = only watch, nothing is saved.
set RECORD=yes
rem With RECORD=yes: also save the clean camera video (yes/no). Needed for training; about 1 GB per hour.
set SAVE_RAW=yes
rem Size of the saved videos: high = the full MODE size (e.g. 1920x1080),
rem   medium = half (960x540, about 1/4 of the disk space), low = a quarter (480x270).
set VIDEO_QUALITY=low
rem Model file. Empty = the model for TARGET:
rem   prawn = prawn_yolo11n_pose_v0.1.pt    fish = fish_tolov26nano.pt
set MODEL=
rem Camera controls. The camera remembers them until it is unplugged.
rem   EXPOSURE: auto, or milliseconds (8, 16, 31, 63 ...)   empty = leave as is
rem   GAIN:     1 = lowest (See3CAM); higher = brighter, noisier   empty = leave as is
rem   WB:       auto, or white balance in Kelvin (e.g. 4500)   empty = leave as is
set EXPOSURE=auto
set GAIN=
set WB=auto
rem ====================================================

if not exist .venv\Scripts\python.exe (
    echo Please double-click install.bat first.
    pause
    exit /b 1
)

set "EXTRA="
if /i not "%RECORD%"=="no" set "EXTRA=--record runs"
if /i "%SAVE_RAW%"=="no" set "EXTRA=%EXTRA% --no-raw"
if defined MODEL set "EXTRA=%EXTRA% --model %MODEL%"
if /i "%PREPROCESS%"=="no" set "EXTRA=%EXTRA% --no-preprocess"
if defined VIDEO_QUALITY set "EXTRA=%EXTRA% --video-quality %VIDEO_QUALITY%"
if defined EXPOSURE set "EXTRA=%EXTRA% --exposure %EXPOSURE%"
if defined GAIN set "EXTRA=%EXTRA% --gain %GAIN%"
if defined WB set "EXTRA=%EXTRA% --wb %WB%"

echo Starting the camera... press q or close the video window to stop, c for camera settings.
.venv\Scripts\python.exe live_prawn_detect.py --source "%CAMERA%" --target %TARGET% --mode %MODE% --kpts %EXTRA%

echo.
if /i "%RECORD%"=="no" (
    echo Done. Nothing was recorded - RECORD=no in run.bat.
    pause
    exit /b 0
)
echo Done. The results are in the "runs" folder (the newest folder is this run).
echo Please zip that folder and send it back.
if exist runs start "" runs
pause
