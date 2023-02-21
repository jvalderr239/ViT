from setuptools import find_packages, setup

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
        "albumentations==1.3.0",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "tqdm==4.64.1"
    ],
    python_requires=">=3.8"
)