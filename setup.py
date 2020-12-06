from setuptools import setup, find_packages

REQUIRED = ["gym", "numpy", "ray[rllib]", "traci", "pandas"]

setup(
    name='utc-reproduction',
    version='0.0',
    packages=['utc-reproduction'],
    install_requires=REQUIRED,
)
