import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepbleed", # Replace with your own username
    version="0.0.1",
    author="msharrock",
    author_email="sharrock@unc.edu",
    description="3D Volumetric Intracerebral Hemorrhage Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/msharrock/deepbleed",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Creative Commons Noncommercial License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.6',
    install_requires=[i.strip() for i in open("requirements.txt").readlines()]
)