import io
import os
import re

from setuptools import find_packages
from setuptools import setup
import versioneer


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    version='0.8',
    name="comet_maths",
    url="https://github.com/comet-toolkit/comet_maths",
    license="LGPLv3",
    author="CoMet Toolkit Team",
    author_email="team@comet-toolkit.org",
    description="Mathematical algorithms and tools to use within CoMet toolkit.",
    long_description=read("README.rst"),
    packages=find_packages(exclude=("tests",)),
    install_requires=["numpy","sklearn","numdifftools","scipy","punpy","matplotlib"],
    extras_require={"dev": ["pre-commit", "tox", "sphinx", "sphinx_rtd_theme"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
