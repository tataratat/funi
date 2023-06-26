#pragma once

#include <iostream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <funi.hpp>

namespace funi {

namespace py = pybind11;

using DefaultIndexType = std::size_t;

template<typename DataType, typename IndexType = DefaultIndexType>
py::tuple Unique(const py::array_t<DataType>& array_2d,
                 const DataType tolerance,
                 const bool stable,
                 const bool return_unique,
                 const bool return_index,
                 const bool sorted_index,
                 const bool return_inverse) {

  // input flag check
  if (!return_unique && !return_index && !return_inverse) {
    throw std::runtime_error(
        "at least one of return_unique, return_index, return_inverse needs to "
        "be True. Given all False.");
  }

  // get shape, get ptr
  const auto array_2d_buf = array_2d.request();
  const DataType* array_2d_ptr = static_cast<DataType*>(array_2d_buf.ptr);
  const IndexType height = static_cast<IndexType>(array_2d_buf.shape[0]);
  const IndexType width = static_cast<IndexType>(array_2d_buf.shape[1]);

  // dim check
  if (array_2d_buf.ndim != 2) {
    throw std::runtime_error("input array must be 2D");
  }

  // prepare unique_ids
  Vector<IndexType> unique_ids, sorted_ids;

  // prepare inverse
  py::array_t<IndexType> inverse;
  IndexType* inverse_ptr = nullptr;
  if (return_inverse) {
    inverse = py::array_t<IndexType>(height);
    inverse_ptr = static_cast<IndexType*>(inverse.request().ptr);
  }

  // compute
  if (stable) {
    UniqueIds<true>(array_2d_ptr,
                    height,
                    width,
                    tolerance,
                    sorted_ids,
                    unique_ids,
                    inverse_ptr);
  } else {
    UniqueIds<false>(array_2d_ptr,
                     height,
                     width,
                     tolerance,
                     sorted_ids,
                     unique_ids,
                     inverse_ptr);
  }
  // get unique count incase we need to return index or data
  const IndexType n_unique = static_cast<IndexType>(unique_ids.size());

  // finally, copy unique_ids
  py::array_t<IndexType> unique_ids_array;
  IndexType* unique_ids_array_ptr = nullptr;
  if (return_index || return_unique) {
    unique_ids_array = py::array_t<IndexType>(n_unique);
    unique_ids_array_ptr =
        static_cast<IndexType*>(unique_ids_array.request().ptr);

    for (IndexType i{}; i < n_unique; ++i) {
      unique_ids_array_ptr[i] = sorted_ids[unique_ids[i]];
    }
  }

  // sorted ids? ok - inverse will be automatically included if needed
  if (sorted_index && return_index) {
    SortIdsAndInverse(n_unique, unique_ids_array_ptr, height, inverse_ptr);
  }

  // prepare copied unique entries
  py::array_t<DataType> unique_data;
  if (return_unique) {
    unique_data = py::array_t<DataType>({n_unique, width});
    DataType* unique_data_ptr =
        static_cast<DataType*>(unique_data.request().ptr);

    for (IndexType i{}; i < n_unique; ++i) {
      std::copy_n(&array_2d_ptr[unique_ids_array_ptr[i] * width],
                  width,
                  &unique_data_ptr[i * width]);
    }
  }

  return py::make_tuple(unique_data, unique_ids_array, inverse);
}

} // namespace funi
