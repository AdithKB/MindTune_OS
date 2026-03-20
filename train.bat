@echo off
REM train.bat — train the EEG classifier (Windows)
REM Run this once after setup.bat, before run.bat

cd /d "%~dp0"

echo.
echo === MindTune-OS — Training Classifier ===
echo.

python src\train_classifier.py
if errorlevel 1 (
    echo.
    echo ERROR: Training failed. Check the error above.
    echo Make sure you have downloaded the EEG dataset to data\eeg_mental_state.csv
    pause
    exit /b 1
)

echo.
echo Training complete. You can now double-click run.bat to start the system.
echo.
pause
