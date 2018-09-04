from setuptools import setup, PEP420PackageFinder

setup(
    name="chmp",
    version="2018.9.1",
    packages=PEP420PackageFinder.find("src"),
    package_dir={"": "src"},
    tests_require=["pytest"],
)
