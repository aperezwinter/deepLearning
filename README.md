## Installation of packages
Packages require for this repository are listed in the file *requirements.txt*.

Follow these steps according to the purpose:
- Create virtual enviroment: `python3 -m venv venv`
- Activate virtual enviroment: `source venv/bin/activate`
- Upgrade pip: `python3 -m pip install --upgrade pip`
- Install packages from *requirements.txt*: `pip install -r requirements.txt`
- Unisntall all packages: `pip freeze | xargs pip uninstall -y`
- Desactivate virtual enviroment: `deactivate`
- Delete virtual enviroment (optional): `rm -rf venv`

Note: Virtual enviroment folder path is aprt of the gitignore file.