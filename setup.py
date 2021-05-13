#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="mdp_extras",
    version="0.0.1",
    install_requires=[
        "numpy",
        "scipy",
        "gym",
        "numba",
    ],
    packages=find_packages(),
)
