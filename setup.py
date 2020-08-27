from setuptools import find_packages, setup


with open('README.md', 'r') as file:
    long_description = file.read()

setup(
    name='hydropt',
    version='0.0.1',
    description='Solves dynamical programming problems for the optimization of hydro power plants.',
    packages=['hydropt'],
    package_dir={'':'src'},
    long_description=long_description,
    long_description_content_type='text/markdown'
)