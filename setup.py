from distutils.core import setup, Extension

setup(
    name='dlib',
    version='0.1.0',
    author='Liam Childs',
    author_email='liam_childs@hotmail.com',
    packages=['dlib'],
    license='LICENSE.txt',
    description='Python bindings of the C++ library dlib',
    long_description=open('README.md').read(),
    ext_modules=[
        Extension('dlib._rvm',
            ['dlib/_rvm.cpp'],
            libraries=['boost_python'],
            extra_compile_args=['-g']
        )
    ] 
)
