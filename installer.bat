@echo off
REM setup_conda_env.bat

REM Set your environment name here
set ENV_NAME=rag_medical_env

REM Create the conda environment with Python 3.9
conda create -y -n %ENV_NAME% python=3.9

REM Activate the environment
call conda activate %ENV_NAME%

REM Install requirements
pip install -r requirements.txt

echo.
echo Environment '%ENV_NAME%' is ready and requirements are installed.
echo To activate later, run: conda activate %ENV_NAME%
pause