from setuptools import setup, find_packages

setup(
    name='rag_chatbot',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.32.0",
        "langchain>=0.1.14",
        "sentence-transformers>=2.2.2",
        "transformers>=4.39.0",
        "torch>=2.1.2",
        "accelerate>=0.28.0",
        "pydantic<2",
        "uvicorn",
        "fastapi",
        "python-dotenv",
        "tqdm",
        "pillow",
        "pymilvus>=2.4.0",
    ],
    author='Your Name',
    description='A RAG-based chatbot using Milvus and Streamlit',
    python_requires='>=3.8',
)
