from setuptools import setup, find_packages

with open("./README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ViT",
    version="0.0.1",
    author="Jose Valderrama",
    author_email="jvalderr239@gmail.com",
    description="Visual Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jvalderr239/ViT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "tqdm",
        "torchvision",
        "torch",
        "requests",
        "torchinfo",
        "opencv-python",
        "matplotlib",
        "rich",
    ],
    python_requires=">=3.8",
    extras_require={
        "dev": ["pylint", "pytest", "pandas"]
    },
)