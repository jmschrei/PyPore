from distutils.core import setup

setup(
    name='PyPore',
    version='0.1.0',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['PyPore'],
    url='http://pypi.python.org/pypi/PyPore/',
    license='LICENSE.txt',
    description='Nanopore Data Analysis package. Provides tools for reading data,\
        performing event detection, segmentation, visualization, and analysis using\
        hidden Markov models, and other tools. Designed for the UCSC Nanopore Group.',
    install_requires=[
        "cython >= 0.20.1",
        "numpy >= 1.8.0",
        "matplotlib >= 1.3.1"
    ],
)
