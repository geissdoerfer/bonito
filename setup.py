from setuptools import setup
from setuptools import find_namespace_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="neslab-bonito",
    version="0.0.1",
    description="Implementation of the Bonito connection protocol for battery-free devices",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Kai Geissdoerfer",
    author_email="kai.geissdoerfer@tu-dresden.de",
    packages=find_namespace_packages(include=["neslab.*"]),
    license="MIT",
    install_requires=["numpy", "scipy", "click"],
    extras_require={"examples": ["matplotlib"]},
    url="https://bonito.nes-lab.org",
    entry_points={"console_scripts": ["pwr2time=neslab.bonito.pwr2time:cli"]},
)
