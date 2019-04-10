import subprocess
import setuptools
from setuptools import setup
import os

if not os.path.exists(os.path.join(os.getcwd(), 'cmake_build')):
    subprocess.check_call('make')

setup(
    name='karma',
    version='0.1',
    author='Pier Fiedorowicz',
    author_email='pierfied@email.arizona.edu',
    description='KARMA (KARMA Algorithm for Reconstructing Mass mAps) - Improved Weak Lensing Mass Maps via a Lognormal Prior',
    long_description='',
    packages=setuptools.find_packages(),
    package_data={'karma': ['*']},
    zip_safe=False,
)
