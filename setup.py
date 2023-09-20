from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Python library for geotechnical calculations"
LONG_DESCRIPTION = "Python library for performing geotechnical engineering calculations"

# Setting up
setup(
    name="geotechpy",
    version=VERSION,
    author="Shawn Hutcherson, P.E.",
    author_email="<hutcherson.shawn@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    readme="README.md",
    packages=find_packages(),
    install_requires=[],
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
