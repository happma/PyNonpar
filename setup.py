from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='PyNonpar',
    version='0.0.1',
    packages=['PyNonpar'],
    url='https://github.com/happma/PyNonpar',
    license='GPL-3',
    author='Martin Happ',
    author_email='martin.happ@aon.at',
    description='Functions and nonparametric test statistics',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy',
        'pandas',
        'pytest',
    ],
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    )
)
