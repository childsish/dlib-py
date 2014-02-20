#include <boost/python.hpp>
#include <vector>
#include <string>
#include <sstream>

#include "errors.h"

using namespace boost::python;

template<typename T>
struct vector_helper
{
    typedef typename T::value_type V;
    
    static std::string str(T& x) {
        std::ostringstream res;
        res << "[";
        if (x.size() > 0) {
            res << x[0];
        }
        for (unsigned int i = 1; i < x.size(); ++i) {
            res << ", " << x[i];
        }
        res << "]";
        return res.str();
    }
    
    static V& get(T& x, int i)
    {
        if( i < 0 ) i += x.size();
        if( i >= 0 && i < (int) x.size() ) return x[i];
        else IndexError();
    }
    
    static void set(T& x, int i, V& v)
    {
        if( i < 0 ) i += x.size();
        if( i >= 0 && i < (int) x.size() ) x[i] = v;
        else IndexError();
    }
    
    static void del(T& x, int i)
    {
        if( i < 0 ) i += x.size();
        if( i >= 0 && i < (int) x.size() ) x.erase(x.begin() + i);
        else IndexError();
    }
};

template<typename T>
void register_vector(const char* type) {
    typedef std::vector<T> vector_type;

    std::string prefix("vector_of_");

    class_<vector_type>(prefix.append(type).c_str())
     .def(init<long>())
     //.def("__init__", &vector_helper<vector_type>::init)
     .def("__str__", &vector_helper<vector_type>::str)
     .def("__repr__", &vector_helper<vector_type>::str)
     .def("__len__", &vector_type::size)
     .def("__iter__", iterator<std::vector<T> >())
     .def("clear", &vector_type::clear)
     .def("append", &vector_type::push_back,
          with_custodian_and_ward<1,2>()) // to let container keep value
     .def("__getitem__", &vector_helper<vector_type>::get,
          return_value_policy<copy_non_const_reference>())
     .def("__setitem__", &vector_helper<vector_type>::set,
          with_custodian_and_ward<1,3>()) // to let container keep value
     .def("__delitem__", &vector_helper<vector_type>::del);
}

