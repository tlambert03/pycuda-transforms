#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding="utf-8").read()


# Add your dependencies here
install_requires = []

install_requires += (["numpy", "pycuda"],)


setup(
    name="pycuda-transforms",
    version="0.1.0",
    author="Talley Lambert",
    author_email="talley.lambert@gmail.com",
    maintainer="Talley Lambert",
    license="BSD-3",
    url="https://github.com/tlambert03/pycuda-transforms",
    description="Affine transforms implemented with pycuda.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    py_modules=["pycuda_transforms"],
    python_requires=">=3.6",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ],
)
