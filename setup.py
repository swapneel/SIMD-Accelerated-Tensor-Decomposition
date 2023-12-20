from setuptools import setup, Extension
import numpy

cosine_distance_module = Extension(
    'cosine_distance',
    sources=['cosine_distance.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-msse2']  # Ensure SSE2 instructions are enabled
)

setup(
    name='cosine_distance',
    version='1.0',
    description='Python interface for the cosine distance C library function using SSE',
    ext_modules=[cosine_distance_module],
)
