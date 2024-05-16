# Code Lib Python

## Install Library

The following command creates a virtual environment, activates it, then install all the required dependency libraries.

``` python
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Virtual Environments

Create virtual environment

``` python
python3 -m venv .venv
```

Activate virtual environment

``` python
source .venv/bin/activate
```

Exit

``` python
deactivate
```

## Python Packages

### Creating / Installing Packages

Create a python package from the *setup.py*, run

``` python
python setup.py sdist bdist_wheel
```

This will create a *dist* directory where the new packge will be created.

Install the python package wheel

``` python
pip install ./path/to/package.whl
```

### Install Packages from Requirements

To install packages from a `requirements.txt` file, run

```bash
pip install -r /path/to/requirement.txt
```

### Save List of Package Dependendcies

1. Create a *requirements.txt* file which contains a list of installed packages and their versions.

    >NOTE: the -all option ensures all the packages are included that might not normally. This includes wheel in order to build the packages, but also pip and setup tools. Not 100% sure of the effect of these when installing, but worked okay on Raspberry Pi.

    ``` python
    pip freeze -all > requirements.txt
    ```

### Upgrading Packages

1. Open the *requirements.txt* and replace `==` (pinned packages) with `>=` (unpinned packages).

2. Upgrade all the unpinned packages.

    ``` bash
    pip install -r requirements.txt --upgrade
    ```

### Resources

[Updating Python Packages](https://www.activestate.com/resources/quick-reads/how-to-update-all-python-packages/)

[Installing Python Packages](https://packaging.python.org/en/latest/tutorials/installing-packages/)

[How To Create a Python Package](https://www.freecodecamp.org/news/build-your-first-python-package/)

[Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

## Tools

A collection of basic tools.

>NOTE: these should be run from the root directory

* **create_package.sh** - creates a python package using *setup.py*
* **clean_build_folders.sh** - removes all the python build folders
* **install_package_wheel.sh** - installs the wheel distribution package
