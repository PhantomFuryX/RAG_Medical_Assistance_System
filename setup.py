from setuptools import setup, find_packages

setup(
    name="medical-assistant",
    version="0.1.0",
    description="A RAG application for medical assistance with LangChain, PyTorch, and ChatOpenAI",
    author="PhantomFuryX",
    author_email="madhabpoulikwork@gmail.com",
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
