import setuptools
import pathlib
from setuptools import find_packages


def get_version() -> str:
    rel_path = "src/ben2/__init__.py"
    with open(rel_path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        req
        for req in f.read().splitlines()
        if req and not (req.startswith("#") or req.strip() is None)
    ]


setuptools.setup(
    name="ben2",
    version=get_version(),
    author="person",
    author_email="email@gmail.com",
    description="A package for BEN2",
    long_description=pathlib.Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/PramaLLC/BEN2",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
