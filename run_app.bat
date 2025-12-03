@echo off
cd /d "%~dp0"
echo ==========================================
echo Resume Search Dashboard Launcher
echo ==========================================

echo 1. Creating Virtual Environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Could not create virtual environment.
    echo Make sure Python is installed and added to PATH.
    pause
    exit /b
)

echo 2. Activating Virtual Environment...
call venv\Scripts\activate

echo 3. Upgrading PIP...
python -m pip install --upgrade pip

echo 4. Installing Dependencies...
echo (This may take a few minutes. We are forcing pre-built binaries.)
pip install -r requirements.txt --only-binary :all:
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Installation failed.
    echo If you see errors about "No matching distribution found",
    echo it means your Python version is too new (e.g. 3.13/3.14) or too old.
    echo Please install Python 3.11.
    pause
    exit /b
)

echo 5. Starting App...
streamlit run app.py

echo.
echo App has closed.
pause
