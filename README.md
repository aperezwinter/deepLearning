## Installation of packages
Packages require for this repository are listed in the file *requirements.txt*.

Follow these steps according to the purpose:

1 On MacOS or Linux systems
- Create virtual environment: `pythonX.XX -m venv venv` replace X.XX for your version (e.g. 3.10)
- Activate virtual environment: `source venv/bin/activate`
- Upgrade pip: `pythonX.XX -m pip install --upgrade pip` replace X.XX for your version (e.g. 3.10) 
- Install packages from *requirements.txt*: `pip install -r requirements.txt`
- Uninstall all packages: `pip freeze | xargs pip uninstall -y`
- Deactivate virtual environment: `deactivate`
- Delete virtual environment (optional): `rm -rf venv`

2 On Windows system
- Create virtual environment: `pythonX.XX -m venv venv` replace X.XX for your version (e.g. 3.10)
- Activate virtual environment: `venv\Scripts\activate`
- Upgrade pip: `pythonX.XX -m pip install --upgrade pip` replace X.XX for your version (e.g. 3.10)
- Install packages from *requirements.txt*: `pip install -r requirements.txt`
- Uninstall all packages: `pip uninstall -r requirements.txt -y`
- Deactivate virtual environment: `deactivate`
- Delete virtual environment (optional): `rmdir /s /q venv`

Note: Virtual environment folder path is part of the gitignore file.