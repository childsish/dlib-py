from distutils.core import setup, Extension

include_dirs = ['/vol/home-vol3/wbi/childsli/opt/include',
    '/vol/home-vol3/wbi/childsli/src/dlib-18.2']

library_dirs = ['/vol/home-vol3/wbi/childsli/opt/lib']

source_files = ['dlib/_rvm.cpp']

setup(
    name='dlib',
    version='0.1.0',
    author='Liam Childs',
    author_email='liam_childs@hotmail.com',
    packages=['dlib'],
    license='LICENSE.txt',
    description='Python bindings of the c++ library dlib',
    long_description=open('README.md').read(),
    ext_modules=[
        Extension('dlib._rvm',
            source_files,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=['boost_python'],
            extra_compile_args=['-g']
        )
    ] 
)
