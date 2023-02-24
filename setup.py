from setuptools import setup


with open("README.md","r") as fh:
    long_description = fh.read()

setup(
    name = 'nndp',
    version = '0.0.1',
    description = 'Dynamic Programming with Neural Networks',
    long_description = long_description,
    long_description_content_type='text/markdown',
    #py_modules = ["dsolve.atoms", "dsolve.expressions", "dsolve.solvers", "dsolve.utils","dsolve.statespace", "dsolve.linearization"],
    package_dir={'':'src'},
    author='Marc de la Barrera i Bardalet',
    url = 'https://github.com/marcdelabarrera/nndp',
    author_email='mbarrera@mit.edu',
    install_requires = ["numpy >=1.20.0"],
    extras_require={"dev":["pytest>=7.1.2",],},
    classifiers =[
        "Programming Language :: Python :: 3.10"
    ]
)