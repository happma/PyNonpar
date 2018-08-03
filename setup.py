from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='PyNonpar',
    version='0.0.2',
    packages=['PyNonpar'],
    url='https://github.com/happma/PyNonpar',
    license='GPL-3',
    author='Martin Happ',
    author_email='martin.happ@aon.at',
    description='Nonparametric Test Statistics',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy',
        'pandas',
        'pytest',
        'scipy',
        'pytest-cov',
        'codecov',
        'numba',
    ],
    classifiers=(
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules"
        "Environment :: Console",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    )
)
