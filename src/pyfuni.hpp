#pragma once

#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "axis.hpp"
#include "lexi.hpp"

namespace funi {

namespace py = pybind11;

using DefaultIndexType = int;

template<typename DataType, typename IndexType = DefaultIndexType>
py::tuple LexiUnique(const py::array_t<DataType>& array_2d,
                     const DataType tolerance,
                     const bool sorted_index) {

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
  py::array_t<IndexType> inverse(height);
  IndexType* inverse_ptr = static_cast<IndexType*>(inverse.request().ptr);

  // compute
  if (sorted_index) {
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
  py::array_t<IndexType> unique_ids_array(n_unique);
  IndexType* unique_ids_array_ptr =
      static_cast<IndexType*>(unique_ids_array.request().ptr);

  for (IndexType i{}; i < n_unique; ++i) {
    unique_ids_array_ptr[i] = sorted_ids[unique_ids[i]];
  }

  // sorted ids? ok - inverse will be automatically included if needed
  if (sorted_index) {
    SortIdsAndInverse(n_unique, unique_ids_array_ptr, height, inverse_ptr);
  }

  // prepare copied unique entries
  py::array_t<DataType> unique_data({n_unique, width});
  DataType* unique_data_ptr = static_cast<DataType*>(unique_data.request().ptr);
  for (IndexType i{}; i < n_unique; ++i) {
    std::copy_n(&array_2d_ptr[unique_ids_array_ptr[i] * width],
                width,
                &unique_data_ptr[i * width]);
  }

  return py::make_tuple(unique_data, unique_ids_array, inverse);
}

/**
 * @brief
 *
 * @param points original points in python numpy format
 * @param tolerance tolerance between two neighboring points to be regarded as
 * one
 * @param stable Preserve order of points
 * @return py::tuple
 */
template<typename DataType, typename IndexType = DefaultIndexType>
py::tuple AxisUnique(const py::array_t<DataType> points,
                     DataType tolerance,
                     bool sorted_index) {
  // Access points
  DataType* p_buf_ptr = static_cast<DataType*>(points.request().ptr);
  IndexType npoints = points.shape(0);
  IndexType pdim = points.shape(1);

  // selecting metric for you.
  std::vector<DataType> metric(pdim, 1.);

  // prepare output arrays
  py::array_t<IndexType> new_indices(npoints);
  IndexType* new_indices_ptr =
      static_cast<IndexType*>(new_indices.request().ptr);

  py::array_t<IndexType> np_inverse(npoints);
  IndexType* inverse = static_cast<IndexType*>(np_inverse.request().ptr);

  // prepare additional input vars
  IndexType nnewpoints{};

  // prepare temp array to store newpoints.
  // we call this thing phil
  py::array_t<DataType> np_newpoints({npoints, pdim});
  DataType* np_newpoints_ptr =
      static_cast<DataType*>(np_newpoints.request().ptr);

  if (sorted_index) {
    Uff<true>(p_buf_ptr,        // original points
              npoints,          // number of original points
              pdim,             // dimensions of points
              metric.data(),    // metric that reorganizes points
              tolerance,        // tolerance between neighboring points
              sorted_index,     // Use sorted_index sort
              np_newpoints_ptr, // pointer to new points (make sure enough space
                                // is allocated)
              new_indices_ptr,  // IDs after resorting
              nnewpoints,       // number of new points
              inverse           // return value, inverse ids to original vector
    );
  } else {
    Uff<false>(p_buf_ptr,
               npoints,
               pdim,
               metric.data(),
               tolerance,
               sorted_index,
               np_newpoints_ptr,

               new_indices_ptr,
               nnewpoints,
               inverse);
  }

  // real newpoints
  assert(nnewpoints > 0);
  np_newpoints.resize({nnewpoints, pdim}, false);
  new_indices.resize({nnewpoints}, false);

  return py::make_tuple(np_newpoints, new_indices, np_inverse);
}

py::tuple UniqueRows(const py::array& array_2d,
                     const double tolerance,
                     const bool sorted_index,
                     const std::string& method) {

  const char dtype = array_2d.dtype().char_();
  const char f = 'f';
  const char d = 'd';

  if (dtype == f) {
    if (method[0] == 'a' || method[0] == 'A') {
      return AxisUnique<float>(array_2d, tolerance, sorted_index);
    } else {
      return LexiUnique<float>(array_2d, tolerance, sorted_index);
    }
  } else if (dtype == d) {

    if (method[0] == 'a' || method[0] == 'A') {
      return AxisUnique<double>(array_2d, tolerance, sorted_index);

    } else {
      return LexiUnique<double>(array_2d, tolerance, sorted_index);
    }
  } else {
    throw std::runtime_error("FUNI supports float32 and float64. For integer "
                             "types, use `np.unique(data, axis=0)`");
  }
  return py::tuple();
}

} // namespace funi
