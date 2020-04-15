# To upload this file to PyPI you must build it then upload it:
# python setup.py sdist bdist_wheel  # build in 'dist' folder
# python-m twine upload dist/*  # 'twine' must be installed: 'pip install twine'


import ast
import io
import re
import os
from setuptools import find_packages, setup

DEPENDENCIES = []
EXCLUDE_FROM_PACKAGES = ["tests*"]
CURDIR = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(CURDIR, "README.md"), "r", encoding="utf-8") as f:
    README = f.read()



setup(
    name="balanced_kmeans",
    version="0.0.6",
    author="@giannisdaras",
    author_email="daras.giannhs@gmail.com",
    description="Balanced kmeans with cuda support in Pytorch.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/giannisdaras/balanced_kmeans",
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    keywords=[],
    scripts=[],
    zip_safe=False,
    install_requires=DEPENDENCIES,
    test_suite="tests.equal_clusters",
    python_requires=">=3.6",
    license="License :: OSI Approved :: MIT License",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
