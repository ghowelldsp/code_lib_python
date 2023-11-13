# enable the virtual environment
source venv/bin/activate

# create the packages
python setup.py sdist bdist_wheel
