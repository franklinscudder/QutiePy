from setuptools import find_packages, setup
setup(
    name='QutiePy',
    packages=find_packages(),
    version='0.1.8',
    long_description='A WIP package providing a simple OOP framework for simulating quantum computing operations in python, with an emphasis on accessability and simplicity.',
    long_description_content_type="text/markdown",
    description='A simple package for learning about and exploring quantum computing.',
    author='T. E. L. Findlay',
    author_email="snick10000@live.co.uk",
    license='MIT',
    scripts=['examples\\BellState.py'],
    classifiers= [
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
        ]
)
