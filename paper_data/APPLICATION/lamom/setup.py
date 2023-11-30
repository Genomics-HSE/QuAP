from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy as np

extension = Extension("calcmom", ["calculate_moments.pyx"],
                language='c', include_dirs=[np.get_include()])

setup(
    name = "calcmom",
    ext_modules = cythonize(extension)
)
