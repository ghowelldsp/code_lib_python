# Code Lib Python

## Enable Virtual Environment

Enter the virtual environment with

``` bash
source venv/bin/activate
```

## Tools

A collection of basic tools.

>NOTE: these should be run from the root directory

* **create_package.sh** - creates a python package using *setup.py*
* **clean_build_folders.sh** - removes all the python build folders
* **install_package_wheel.sh** - installs the wheel distribution package

## Package Python Library

Create a python package from the *setup.py*, run

``` bash
python setup.py sdist bdist_wheel
```

This will create a *dist* directory where the new packge will be created.

Install the python package wheel

``` bash
pip install ./path/to/package.whl
```

### Resources

[Installing Python Packages](https://packaging.python.org/en/latest/tutorials/installing-packages/)

[How To Create a Python Package](https://www.freecodecamp.org/news/build-your-first-python-package/)

[Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

## Updating Python Packages

To update the python packages used follow the instructions below.

1. Create a *requirements.txt* file which contains a list of installed packages and their versions.

    ``` bash
    pip freeze > requirements.txt
    ```

2. Open the *requirements.txt* and replace `==` (pinned packages) with `>=` (unpinned packages).

3. Upgrade all the unpinned packages.

    ``` bash
    pip install -r requirements.txt --upgrade
    ```

### Resources

[Updating Python Packages](https://www.activestate.com/resources/quick-reads/how-to-update-all-python-packages/)
