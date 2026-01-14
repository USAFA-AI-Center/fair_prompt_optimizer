from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read() if __import__("os").path.exists("README.md") else ""

setup(
    name="fair_prompt_optimizer",
    version="0.2.0",
    description="DSPy-powered prompt optimization for FAIR-LLM agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="USAFA AI Center",
    url="https://github.com/usafa-ai-center/fair-prompt-optimizer",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "dspy-ai>=2.0.0",
        "nest-asyncio>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fair-optimize=fair_prompt_optimizer.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)