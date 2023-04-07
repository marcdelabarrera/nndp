from setuptools import setup


with open("README.md","r") as fh:
    long_description = fh.read()

setup(
    name = 'nndp',
    version = '0.0.5',
    description = 'Dynamic Programming with Neural Networks',
    long_description = long_description,
    long_description_content_type='text/markdown',
    py_modules = ["nndp.policy_function", "nndp.core"],
    package_dir={'':'src'},
    author='Marc de la Barrera i Bardalet',
    url = 'https://github.com/marcdelabarrera/nndp',
    author_email='mbarrera@mit.edu',
    install_requires = ["numpy >= 1.20.0","jax >= 0.3.22", "optax >= 0.1.0", "jaxlib >= 0.3.22", "dm-haiku >= 0.0.9"],
    extras_require={"dev":["pytest >= 7.1.2",]},
    classifiers =[
        "Programming Language :: Python :: 3.10"
    ]
)