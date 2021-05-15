import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pydrm-Mallory.Wittwer",
    version="0.0.1",
    author="Mallory Wittwer",
    author_email="mallory.wittwer@gmail.com",
    description="A package for the analysis of DRM data using Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MalloryWittwer/pydrm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)