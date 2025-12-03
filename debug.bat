@echo off
cd /d "%~dp0"
echo ========================================================
echo               DEBUG / DIAGNOSTIC TOOL
echo ========================================================
echo.
echo 1. CHECKING DIRECTORY...
echo Current Directory: "%CD%"
echo.
echo Files found here:
dir /b
echo.
echo --------------------------------------------------------
echo.
echo 2. CHECKING PYTHON...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
)
echo.
echo --------------------------------------------------------
echo.
echo 3. CHECKING REQUIREMENTS FILE...
if exist requirements.txt (
    echo [OK] requirements.txt found.
) else (
    echo [FAIL] requirements.txt NOT found!
    echo.
    echo SOLUTION:
    echo You are likely running this from inside the ZIP file.
    echo Please right-click the zip, select "Extract All", 
    echo and run this script from the extracted folder.
)
echo.
echo --------------------------------------------------------
echo.
echo 4. TEST INSTALLATION...
echo Creating temporary test environment...
python -m venv venv_debug
call venv_debug\Scripts\activate

echo.
echo Attempting to install streamlit...
pip install streamlit
if %errorlevel% neq 0 (
    echo [FAIL] Could not install streamlit.
    echo This might be a network issue or missing C++ build tools.
) else (
    echo [OK] Streamlit installed successfully.
)

echo.
echo ========================================================
echo DIAGNOSTIC COMPLETE. PLEASE SEND A PHOTO OF THIS SCREEN.
echo ========================================================
pause
