"""Setup.
"""
from setuptools import find_packages
from setuptools import setup

setup(
    name="AstroPF",
    version="0.0.1",
    description="Xspec inference with particle filter",
    author="Tomoki Omama",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "tensorflow_probability"
    ],
    classifiers=[
        "Development Status :: 1 - Planning"
    ],
)
