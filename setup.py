from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirements  = []
    with open(file_path) as file:
        file_data = file.readlines()
        requirements=file_data
        r = []
        for res in requirements:
            r.append(res.replace('\n',''))
        
        if  '-e .' in r:
            r.remove('-e .')
    return r

setup(
    name='ML_Project',
    version='0.1.0',
    url='https://github.com/bilalrana8718/ML_Project',
    author='Rana Bilal Akbar',
    author_email='bilal.rana8718@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)