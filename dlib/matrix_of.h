#include <boost/python.hpp>
#include <dlib/matrix.h>
#include <string>
#include <sstream>

#include "errors.h"

using namespace boost::python;

template<typename T>
struct matrix_helper
{
    typedef typename T::value_type V;
    
    static std::string str(const T& x) {
        if (x.size() == 0) return "[[]]";
        
        std::ostringstream res;
        res << "[[" << x(0);
        long i = 1;
        while (i < x.size()) {
            if (i % x.nc() == 0) {
                res << "]";
                if (i / x.nc() < x.nr()) {
                    res << std::endl << " [" << x(i);
                    i++;
                }
            }
            res << ", " << x(i);
            i++;
        }
        res << "]";
        return res.str();
    }
    
    static V& get(T& x, const tuple& key) {
        long row = extract<long>(key[0]);
        long col = extract<long>(key[1]);
        
        if ( row < 0 ) row += x.nr();
        if ( col < 0 ) col += x.nc();
        if ( row >= x.nr() ) IndexError();
        if ( col >= x.nc() ) IndexError();
        return x(row, col);
    }
    
    static void set(T& x, const tuple& key, V& v) {
        long row = extract<long>(key[0]);
        long col = extract<long>(key[1]);
        
        if ( row < 0 ) row += x.nr();
        if ( col < 0 ) col += x.nc();
        if ( row >= x.nr() ) IndexError();
        if ( col >= x.nc() ) IndexError();
        x(row, col) = v;
    }
    
    static const tuple get_size(const T& x) {
       return make_tuple(x.nr(), x.nc());
    }

    static void set_size(T& x, const tuple& shape) {
        x.set_size(extract<long>(shape[0]), extract<long>(shape[1]));
    }
};

template<typename T>
void register_matrix(const char* type) {
    typedef dlib::matrix<T> matrix_type;

    std::string prefix("matrix_of_");

    class_<matrix_type>(prefix.append(type).c_str())
     .def(init<long, long>())
     .def("__str__", &matrix_helper<matrix_type>::str)
     .def("__repr__", &matrix_helper<matrix_type>::str)
     .def("__getitem__", &matrix_helper<matrix_type>::get,
         return_value_policy<copy_non_const_reference>())
     .def("__setitem__", &matrix_helper<matrix_type>::set,
          with_custodian_and_ward<1,3>()) // to let container keep value
     .add_property("shape",
        &matrix_helper<matrix_type>::get_size,
        &matrix_helper<matrix_type>::set_size);
}

