from setuptools import setup


with open("README.md","r") as fh:
    long_description = fh.read()

setup(
    name = 'dsolve',
    version = '0.0.9',
    description = 'Solver of dynamic equations with forward looking variables',
    long_description = long_description,
    long_description_content_type='text/markdown',
    py_modules = ["dsolve.atoms", "dsolve.expressions", "dsolve.solvers", "dsolve.utils","dsolve.statespace", "dsolve.linearization"],
    package_dir={'':'src'},
    author='Marc de la Barrera i Bardalet',
    url = 'https://github.com/marcdelabarrera/dsolve',
    author_email='mbarrera@mit.edu',
    install_requires = ["scipy >= 1.9.0", "sympy >= 1.11", "numpy >=1.20.0", "IPython >= 7.12.0"],
    extras_require={"dev":["pytest>=7.1.2",],},
    classifiers =[
        "Programming Language :: Python :: 3.10"
    ]
)