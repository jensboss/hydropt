from setuptools import find_packages, setup


with open('README.md', 'r') as file:
    long_description = file.read()

setup(
    name='hydropt',
    version='0.0.2',
    author="Jens Boss",
    author_email="bossjens@gmail.com",
    description='Solves dynamical programming problems for the optimization of hydro power plants.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/yenzmike/hydropt",
    packages=['hydropt'],
    package_dir={'':'src'},
    include_package_data=True,
    package_data={'': ['data/*.csv']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "pandas>=0.25.3",
        "numpy>=1.17.4",
        "scipy>=1.5.0"
    ],
    extras_require={
        "dev": [
            "pytest>=5.4.2",
        ],
    },
)