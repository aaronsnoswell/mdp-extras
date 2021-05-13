#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="mdp_extras",
    version="0.0.2",
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "gym",
        "numba",
    ],
    packages=find_packages(),
)
