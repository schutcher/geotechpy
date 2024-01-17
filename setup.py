from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

VERSION = "0.0.4"
DESCRIPTION = "Python library for geotechnical calculations"
LONG_DESCRIPTION = "Python library for performing geotechnical engineering calculations"

# Setting up
setup(
    name="geotechpy",
    version=VERSION,
    author="Shawn Hutcherson, P.E.",
    author_email="hutcherson.shawn@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["matplotlib>=3.8.2", "numpy>=1.26.2", "pandas>=2.1.4"],
    keywords=["python", "geotechpy", "geotechnical", "geotech"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering",
        "Natural Language :: English",
    ],
)
