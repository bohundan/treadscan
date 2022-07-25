import os

import numpy as np
from setuptools import find_packages, setup
from distutils.extension import Extension
from Cython.Build import cythonize


os.environ['CFLAGS'] = '-shared -pthread -fPIC -fwrapv -O3 -fno-strict-aliasing -fopenmp -I "/usr/include/python3.8"'


with open('README.md', 'r') as f:
    long_description = f.read()



setup(
    name='treadscan',
    version='1.0.2',
    package_dir={'': 'src'},
    packages=find_packages(),
    author='Daniel Bohuněk',
    description='Tools for scanning tire treads.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/bohundan/treadscan',
    setup_requires=['cython'],
    ext_modules=cythonize(Extension('treadscan.cwrap', ['src/treadscan/cwrap.pyx']), compiler_directives={'language_level': 3}),
    install_requires=[
        'improutils',
        'numpy',
        'opencv-python',
        'opencv-contrib-python',
        'torch',
        'torchvision'
    ]
)
