#ifndef ERRORS_H
#define ERRORS_H

void IndexError() {
    PyErr_SetString(PyExc_IndexError, "Index out of range");
}

#endif

