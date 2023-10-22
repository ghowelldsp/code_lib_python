from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Code Lib Python'
LONG_DESCRIPTION = 'Python Code Library'

setup(
        name="codelibpython", 
        version=VERSION,
        author="George Howell",
        author_email="<georgehowelldsp@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],
        keywords=['python'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ]
)