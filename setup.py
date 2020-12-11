from setuptools import setup

REQUIRED = ["gym", "numpy", "pandas", "traci", "ray[rllib]", "matplotlib"]

setup(
    name='utc_reproduction',
    version='0.0',
    packages=['utc_reproduction'],
    install_requires=REQUIRED,
)
