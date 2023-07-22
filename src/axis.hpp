#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

/// @brief
namespace funi {

/* util routines */
template <typename T, typename IndexingType>
inline T DiffNormSquared(const T* a, const IndexingType& start_a,
                         const IndexingType& start_b,
                         const IndexingType& size) {
  T res{};
  for (IndexingType i{}; i < size; i++) {
    res +=
        (a[start_a + i] - a[start_b + i]) * (a[start_a + i] - a[start_b + i]);
  }
  return res;
}

/*
 * Sort Vector using lambda expressions
 * ref:
 *   stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
 */
template <typename IndexingType, typename T>
std::vector<IndexingType> ArgSort(const std::vector<T>& v) {
  std::vector<IndexingType> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  std::stable_sort(
      idx.begin(), idx.end(),
      [&v](IndexingType i1, IndexingType i2) { return v[i1] < v[i2]; });

  return idx;
}

template <bool stable_sort, typename DataType, typename IndexingType>
inline void Uff(DataType* original_points,          /* in */
                IndexingType& number_of_points,     /* in */
                IndexingType& point_dim,            /* in */
                DataType* metric,                   /* in */
                DataType& tolerance,                /* in */
                const bool& stable,                 /* in */
                DataType* new_points,               /* out */
                IndexingType* new_indices,          /* out */
                IndexingType& number_of_new_points, /* out */
                IndexingType* inverse) {            /* out */

  const DataType tolerance_squared{tolerance * tolerance};

  // Create a vector that contains the metric
  std::vector<DataType> vector_metric{};
  vector_metric.resize(number_of_points);
  for (IndexingType i{0}; i < number_of_points; i++) {
    vector_metric[i] = metric[0] * original_points[i * point_dim];
    for (IndexingType j{1}; j < point_dim; j++) {
      vector_metric[i] += metric[j] * original_points[i * point_dim + j];
    }
  }

  // Sort Metric Vector
  const auto metric_order_indices = ArgSort<IndexingType>(vector_metric);

  // Reallocate new vector, set to -1 to mark untouched
  std::vector<IndexingType> stable_inverse;
  std::vector<bool> newpointmasks(number_of_points);  // zero (false) init
  std::fill(inverse, inverse + number_of_points, -1);

  // Loop over points
  number_of_new_points = 0;
  for (IndexingType lower_limit{0};
       lower_limit < static_cast<IndexingType>(metric_order_indices.size()) - 1;
       lower_limit++) {
    // PoIndexingType already processed
    if (inverse[metric_order_indices[lower_limit]] != -1) {
      continue;
    } else {
      newpointmasks[metric_order_indices[lower_limit]] = true;
    }

    // Value only required for stable sort (tracks the lowest occurence id of a
    // given point, that might be duplicate)
    IndexingType current_lowest_id{metric_order_indices[lower_limit]};

    // PoIndexingType has not been processed -> add it to new point
    // list
    if (!stable) {
      for (IndexingType i_dim{0}; i_dim < point_dim; i_dim++) {
        new_points[number_of_new_points * point_dim + i_dim] =
            original_points[metric_order_indices[lower_limit] * point_dim +
                            i_dim];
        new_indices[number_of_new_points] = metric_order_indices[lower_limit];
      }
    }
    inverse[metric_order_indices[lower_limit]] = number_of_new_points;

    // Now check allowed range for duplicates
    IndexingType upper_limit = lower_limit + 1;
    while ((vector_metric[metric_order_indices[upper_limit]] -
            vector_metric[metric_order_indices[lower_limit]]) < tolerance) {
      const bool is_duplicate =
          DiffNormSquared(original_points,
                          metric_order_indices[lower_limit] * point_dim,
                          metric_order_indices[upper_limit] * point_dim,
                          point_dim) < tolerance_squared;
      if (is_duplicate) {
        inverse[metric_order_indices[upper_limit]] = number_of_new_points;
        newpointmasks[metric_order_indices[upper_limit]] = false;
        // If stable, the index with the lower id needs to be stored
        if ((stable) &&
            (metric_order_indices[upper_limit] < current_lowest_id)) {
          newpointmasks[metric_order_indices[upper_limit]] = true;
          newpointmasks[current_lowest_id] = false;
          current_lowest_id = metric_order_indices[upper_limit];
        }
      }
      upper_limit++;
      if (upper_limit >=
          static_cast<IndexingType>(metric_order_indices.size())) {
        break;
      }
    }
    number_of_new_points++;
  }

  // Special case
  const auto& last_index = metric_order_indices.size() - 1;
  if (inverse[metric_order_indices[last_index]] == -1) {
    if (!stable) {
      for (IndexingType i_dim{0}; i_dim < point_dim; i_dim++) {
        new_points[number_of_new_points * point_dim + i_dim] =
            original_points[metric_order_indices[last_index] * point_dim +
                            i_dim];
        new_indices[number_of_new_points] = metric_order_indices[last_index];
      }
    }
    inverse[metric_order_indices[last_index]] = number_of_new_points;
    number_of_new_points++;
    newpointmasks[metric_order_indices[last_index]] = true;
  }

  if (stable) {
    IndexingType counter{};
    // This could be a map (but I think this is sufficient for the moment,
    // especially if only a few number of duplicates exist)
    stable_inverse.assign(number_of_points, -1);
    for (IndexingType i{0}; i < number_of_points; i++) {
      if (newpointmasks[i]) {
        for (IndexingType j{0}; j < point_dim; j++) {
          new_points[counter * point_dim + j] =
              original_points[i * point_dim + j];

          new_indices[counter] = i;
        }
        stable_inverse[inverse[i]] = counter;
        counter++;
      }
      inverse[i] = stable_inverse[inverse[i]];
    }
  }
}

}  // namespace funi
