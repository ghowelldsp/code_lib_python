# Code Lib Python

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
