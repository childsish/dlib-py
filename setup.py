from distutils.core import setup, Extension

setup(
    name='dlib',
    version='0.1.0',
    author='Liam Childs',
    author_email='liam_childs@hotmail.com',
    packages=['dlib', 'dlib.test'],
    license='LICENSE.txt',
    description='Python bindings of the c++ library dlib',
    long_description=open('README.txt').read(),
    ext_modules=[
        Extension('dlib.rvm',
            ['dlib/rvm_binding.cpp'],
            include_dirs=['aSimpleExample'],
            library_dirs=['/'],
            libraries=['boost_python'],
            extra_compile_args=['-g']
        )
    ] 
)