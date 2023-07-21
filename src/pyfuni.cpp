#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pyfuni.hpp>

PYBIND11_MODULE(funi, m) { m.def("unique_rows", &funi::UniqueRows); }
