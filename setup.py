from typing import List

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def read_requirements(file: str) -> List[str]:
    """Read requirements files."""
    out = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                if line.startswith("-r"):
                    folder = "/".join(file.split("/")[:-1])
                    out += read_requirements(f"{folder}/{line.split(' ')[-1].strip()}")
                else:
                    out.append(line.strip())
    return out


install_requires = read_requirements("requirements/requirements.txt")
dev_requires = read_requirements("requirements/requirements_dev.txt")

# Debug print statements. Used with python setup.py -v develop
print(f"install_requires: {install_requires}")
print(f"dev_requires: {dev_requires}")

setup(
    name="pika",
    version="0.1.0",
    author="EMCarrami",
    author_email="eli.carrami@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require={"dev": dev_requires},
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
