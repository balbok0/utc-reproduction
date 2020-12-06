from setuptools import setup, find_packages

REQUIRED = ["gym", "pandas", "traci", "ray[rllib]"]

setup(
    name='utc_reproduction',
    version='0.0',
    packages=['utc_reproduction'],
    install_requires=REQUIRED,
)
