#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pyfuni.hpp>

PYBIND11_MODULE(funi, m) {
  m.def("unique_rows",
        &funi::UniqueRows,
        pybind11::arg("query"),
        pybind11::arg("tolerance"),
        pybind11::arg("sorted_index"),
        pybind11::arg("method"));
}
