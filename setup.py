from setuptools import find_packages, setup
from typing import List

# create our project as a package and also deployed on pypi

hypen_e_dot = '-e .'
def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of requirements
    -e . in requirements.txt file automaticly maps setup.py file
    """
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
    
    if hypen_e_dot in requirements:
        requirements.remove(hypen_e_dot)
    return requirements



setup(
    name='ML_Project',
    version='0.0.1',
    author='Sumit Dhakad',
    author_email='sumit.dhakad9644@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')

)

