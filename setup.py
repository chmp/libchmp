from setuptools import setup, PEP420PackageFinder


setup(
    name="chmp",
    version="21.2.0",
    description="Support code for machine learning / data science experiments",
    author="Christopher Prohm",
    long_description=open("Readme.pypi.md").read(),
    long_description_content_type="text/markdown",
    packages=PEP420PackageFinder.find("src"),
    package_dir={"": "src"},
    tests_require=["pytest"],
    url="https://github.com/chmp/misc-exp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
