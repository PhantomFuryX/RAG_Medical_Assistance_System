import os

# Define folder structure
folders = [
    "docs",
    "src/data",
    "src/models",
    "src/nlp",
    "src/integration",
    "src/utils",
    "tests",
    ".github/workflows",
]

# Define files and their initial content
files = {
    "Dockerfile": "# Dockerfile for medical assistant application\n",
    "docker-compose.yml": "# docker-compose file for medical assistant application\n",
    "requirements.txt": (
        "openai>=0.27.0\n"
        "langchain>=0.0.140\n"
        "torch>=1.13.0\n"
        "pymongo>=4.0.0\n"
        "fastapi>=0.78.0\n"
        "uvicorn[standard]>=0.18.0\n"
        "python-dotenv>=0.21.0\n"
        "pytest>=7.0.0\n"
    ),
    "README.md": "# Medical Assistant Application\n\nProject documentation and overview.\n",
    "setup.py": '''from setuptools import setup, find_packages

setup(
    name="medical-assistant",
    version="0.1.0",
    description="A RAG application for medical assistance with LangChain, PyTorch, and ChatOpenAI",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai>=0.27.0",
        "langchain>=0.0.140",
        "torch>=1.13.0",
        "pymongo>=4.0.0",
        "fastapi>=0.78.0",
        "uvicorn[standard]>=0.18.0",
        "python-dotenv>=0.21.0",
    ],
    entry_points={{
        "console_scripts": [
            "medical-assistant=template:main",
        ],
    }},
)
''',
    ".github/workflows/ci.yml": '''name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Tests
        run: |
          pytest --maxfail=1 --disable-warnings -q

      - name: Build Docker Image
        run: |
          docker build -t medical-assistant:latest .
'''
}

def create_folders():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")
        # Create an empty __init__.py file in each folder
        init_file = os.path.join(folder, "__init__.py")
        with open(init_file, "w", encoding="utf-8") as f:
            pass
        print(f"Created file: {init_file}")

def create_files():
    for filepath, content in files.items():
        # Ensure the directory exists for the file
        dirpath = os.path.dirname(filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created file: {filepath}")

def main():
    create_folders()
    create_files()
    print("Project structure has been created successfully.")

if __name__ == "__main__":
    main()
