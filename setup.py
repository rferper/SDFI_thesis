from setuptools import setup

setup(
    name='SDFI_thesis',
    version='1.1',
    packages=['SDFI'],
    url='https://github.com/rferper/',
    license='mit',
    author='RaquelFernandezPeralta',
    author_email='r.fernandez@uib.es',
    description='Subgroup discovery based on fuzzy implication functions',
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "random",
        "pandas",
        "pathlib",
        "progressbar",
        "copy",
        "math",
        "pyFTS",
        "matplotlib"
    ]
)
