dlib-py
=======
This project is ultimately intended to provide python bindings for the excellent C++ library dlib by David E. King ([url][1], [article][2]). Currently only an RVM module has been implemented (that's what I'm interested in). This has been done in a mixture of C++ and Python due to the heavy use of templates in the dlib C++ Library. In future, I'll look for ways to compile directly to a more Pythonic module without needing any Python code.

Dependencies
------------
 * Boost::Python
 * dlib

Installation
------------
 1. Install Boost::Python using a package manager or from source. Make sure you've built the dynamic library (libboost_python.so)
 2. Unzip the dlib headers to the directory where you keep your sources folders.
 3. Install dlib-py:
  * In the directory where you keep your source folders (change the necessary directories to something suitable
   ```bash
   git clone git@github.com:childsish/dlib-py.git 
   cd dlib-py
   export CFLAGS="-I/path/to/dlib"
   python setup.py install
   ```

Hints
-----
Make sure all the necessary shared libraries are in LD_LIBRARY_PATH

Notes
-----
There are some strange programming hacks. I hope to create a list here for future inspection:
 * When depickling functions, I try each possible function and return the successful one. Otherwise I throw an exception. In future I should implement proper pickling and depickling to make the module more Pythonic
 * Not all bindings could be declared in the helper functions. For example, declaring the train function of the trainers, results in a syntax error in a helper function, but will work in the main function. Thus I do as much as possible in the helper functions, but some things are still done in the main function.
 * There are still some warnings when compiling the bindings.
 * There's probably some more strange stuff that I did but can't remember at the moment.

Of course, the code should be properly documented too.

[1]: http://dlib.net/ "Dlib-ml: A Machine Learning Toolkit."
[2]: http://jmlr.org/papers/volume10/king09a/king09a.pdf "Davis E. King. Dlib-ml: A Machine Learning Toolkit. Journal of Machine Learning Research 10, pp. 1755-1758, 2009 "
