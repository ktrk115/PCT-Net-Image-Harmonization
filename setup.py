from setuptools import setup, find_packages

setup(
    name="iharm",
    version="0.1.0",
    packages=find_packages(include=["iharm", "iharm.*"]),
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "einops",
    ],
)
