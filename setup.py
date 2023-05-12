from setuptools import setup
import setuptools

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="rl_squared",
    install_requires=required,
    packages=setuptools.find_packages(
        exclude=[
            "experiments",
            "notes",
        ]
    ),
)
