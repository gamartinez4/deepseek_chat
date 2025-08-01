from setuptools import setup, find_packages

setup(
    name="product-rag-bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "langchain>=0.1.16",
        "langgraph>=0.0.19",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "pydantic>=2.5.3",
        "pytest>=7.4.4",
        "pydantic-settings>=2.1.0",
        "langchain-huggingface>=0.0.1",
        "typing_extensions>=4.7.0",
    ],
) 