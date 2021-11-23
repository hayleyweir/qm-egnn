#!/usr/bin/env python

"""The setup script."""

from setuptools import setup
import glob
import os

setup(
    author="HW,SH,MM",
    author_email="spam@hotmail.com",
    python_requires=">=3.5",
    description="Run qm_egnn",
    name="qm_egnn",
    packages=["src"],
    scripts=["main.py"]
)
