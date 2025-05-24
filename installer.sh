#!/bin/bash

# Set your environment name here
ENV_NAME="rag_medical_env"

# Create the conda environment with Python 3.9
conda create -y -n $ENV_NAME python=3.9

# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install requirements
pip install -r requirements.txt

echo
echo "Environment '$ENV_NAME' is ready and requirements are installed."
echo "To activate later, run: conda activate $ENV_NAME"