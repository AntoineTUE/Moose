import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Moose",
    version="0.0.1",
    author="Antoine Salden",
    author_email="toine.salden@unitn.it",
    description="A thin wrapper around MassiveOES to extend functionality and streamline workflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/pioneerco/analysis/moose",
    project_urls={
        "Bug Tracker": "https://gitlab.com/pioneerco/analysis/moose/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "Moose"},
    packages=setuptools.find_packages(where="./Moose"),
    package_data = {"data":["*.db"]},
    python_requires=">=3.9",
    install_requires = ['pandas', 'numpy', 'matplotlib', 'lmfit', 'scipy']
)