from setuptools import find_packages, setup

setup(
    name="dynprog",
    version="0.0.1",
    description="Solve dynamical programming problems for the optimization of hydro power plants.",
    py_modules=["model","senario","dynprog"],
    package_dir={'':'src'}
)