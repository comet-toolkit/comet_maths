import io
import os
import re

from setuptools import find_packages
from setuptools import setup

exec(open("comet_maths/_version.py").read())


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type("")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    version=__version__,
    name="comet_maths",
    url="https://github.com/comet-toolkit/comet_maths",
    license="LGPLv3",
    author="CoMet Toolkit Team",
    author_email="team@comet-toolkit.org",
    description="Mathematical algorithms and tools to use within CoMet toolkit.",
    long_description=read("README.rst"),
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "numpy",
        "scikit-learn",
        "numdifftools",
        "scipy",
        "punpy",
        "matplotlib",
        "obsarray",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "tox",
            "sphinx",
            "sphinx_design",
            "sphinx_book_theme",
            "ipython",
            "sphinx_autosummary_accessors",
        ],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
