from importlib.metadata import entry_points
from setuptools import setup

setup(
    name="uster_processing",
    version="1.2.0",
    description="Data processing tool for USTER.",
    url='',
    author='Heath Smith',
    author_email="hsmith@thorlabs.com",
    license="Not Licensed",
    packages=['uster_processing'],
    entry_points={
        'console_scripts': [
            'uster=uster_processing.main:main'
        ]
    },
    zip_safe=False
)