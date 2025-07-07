from setuptools import setup, find_packages

setup(
    name="balatro-controllers",
    version="0.1.0",
    description="Python controllers for interacting with Balatro game instances",
    author="Harry Liuson",
    author_email="",
    packages=find_packages(),
    py_modules=["controller"],
    python_requires=">=3.7",
    install_requires=[
        # Add any dependencies your controller.py needs
        # Based on the imports, these appear to be standard library modules
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="balatro game controller automation",
    project_urls={
        "Source": "https://github.com/your-repo/balatro-agent",
    },
)

