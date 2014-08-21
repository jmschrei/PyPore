from distutils.core import setup
from distutils.extension import Extension
import numpy as np

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }

if use_cython:
    ext_modules = [
        Extension("PyPore.cparsers", [ "PyPore/cparsers.pyx" ], include_dirs=[np.get_include()] ),
        Extension("PyPore.calignment", [ "PyPore/calignment.pyx" ], include_dirs=[np.get_include()] )
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules = [
        Extension("PyPore.cparsers", [ "PyPore/cparsers.c" ], include_dirs=[np.get_include()] ),
        Extension("PyPore.calignment", [ "PyPore/calignment.c" ], include_dirs=[np.get_include()] )
    ]

setup(
    name='pythonic-porin',
    version='0.2.0',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['PyPore'],
    url='http://pypi.python.org/pypi/pythonic-porin/',
    license='LICENSE.txt',
    description='Nanopore Data Analysis package. Provides tools for reading data,\
        performing event detection, segmentation, visualization, and analysis using\
        hidden Markov models, and other tools. Designed for the UCSC Nanopore Group.',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    install_requires=[
        "cython >= 0.20.1",
        "numpy >= 1.8.0",
        "matplotlib >= 1.3.1"
    ],
)
