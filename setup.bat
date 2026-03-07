@echo off
REM setup.bat — first-time Windows setup for MindTune-OS
REM Run this once before using run.bat

cd /d "%~dp0"

echo.
echo === MindTune-OS First-Time Setup ===
echo.

REM ── Check Python ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not on PATH.
    echo        Download Python from https://www.python.org/downloads/
    echo        Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM ── Install dependencies ──────────────────────────────────────────────────
echo Installing Python dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Check the error above.
    pause
    exit /b 1
)
echo.

REM ── Create .env if it doesn't exist ──────────────────────────────────────
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo Created .env from .env.example
    ) else (
        echo WARNING: .env.example not found. Create .env manually.
    )
) else (
    echo .env already exists — skipping copy.
)

echo.
echo Setup complete!
echo.
echo Next steps:
echo   1. Open .env in a text editor and fill in your API keys.
echo   2. Double-click run.bat to start the system.
echo.
pause
