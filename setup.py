import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GeneticProgrammingAfpo",
    version="0.0.2",
    author="Ryan Grindle",
    author_email="ryan.grindle@uvm.edu",
    description="Implementation of Genetic Programming.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)