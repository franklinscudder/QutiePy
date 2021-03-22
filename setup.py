from setuptools import find_packages, setup
import os

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = [] 

if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='QutiePy',
    py_modules=['qutiepy'],
    version='0.1.17',
    long_description='A WIP package providing a simple OOP framework for simulating quantum computing operations in python, with an emphasis on accessability and simplicity.',
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    description='A simple package for learning about and exploring quantum computing.',
    author='T. E. L. Findlay',
    author_email="snick10000@live.co.uk",
    license='MIT',
    classifiers= [
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
        ]
)
