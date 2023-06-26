#pragma once

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace funi {

/// http://stackoverflow.com/a/21028912/273767
template<typename Type, typename BaseAllocator = std::allocator<Type>>
class DefaultInitializationAllocator : public BaseAllocator {
  using AllocatorTraits_ = std::allocator_traits<BaseAllocator>;

public:
  template<typename U>
  struct rebind {
    using other = DefaultInitializationAllocator<
        U,
        typename AllocatorTraits_::template rebind_alloc<U>>;
  };

  using BaseAllocator::BaseAllocator;

  template<typename U>
  static void
  construct(U* ptr) noexcept(std::is_nothrow_default_constructible<U>::value) {
    ::new (static_cast<void*>(ptr)) U;
  }
  template<typename U, typename... Args>
  void construct(U* ptr, Args&&... args) {
    AllocatorTraits_::construct(static_cast<BaseAllocator&>(*this),
                                ptr,
                                std::forward<Args>(args)...);
  }
};

template<typename Type>
using Vector = std::vector<Type, DefaultInitializationAllocator<Type>>;

namespace internal {

/// ArgSort along the height expects sorted_ids to be np.arnage(height * width)
template<bool stable_sort = false, typename DataType, typename IndexType>
void ArgSortAlongHeight(const DataType* to_sort,
                        const IndexType height,
                        const IndexType width,
                        const DataType tolerance,
                        Vector<IndexType>& sorted_ids) {
  // minimal size check
  if (static_cast<IndexType>(sorted_ids.size()) != height) {
    throw std::runtime_error("internal::ArgSortAlongHeight - input sorted_ids "
                             "does not match size of arrays to be sorted.");
  }

  // we assume the same width
  auto lexicographical_compare = [&](const IndexType& i_a,
                                     const IndexType& i_b) -> bool {
    // get a's beginning ptr  and its end
    const DataType* a_ptr = &to_sort[i_a * width];
    const DataType* a_end = a_ptr + width;

    // get b's begging ptr
    const DataType* b_ptr = &to_sort[i_b * width];

    // compare
    for (; a_ptr != a_end; ++a_ptr, ++b_ptr) {
      // get diff
      const DataType ab_diff = *a_ptr - *b_ptr;

      // same within the tolerance?
      if (std::abs(ab_diff) < tolerance) {
        continue;
        // not the same.
      } else {
        return ab_diff < 0;
      }
    }
    // same. return false.
    return false;
  };

  // sort
  if constexpr (stable_sort) {
    std::stable_sort(sorted_ids.begin(),
                     sorted_ids.end(),
                     lexicographical_compare);
  } else {
    std::sort(sorted_ids.begin(), sorted_ids.end(), lexicographical_compare);
  }
}

} // namespace internal

template<bool stable_sort = false, typename DataType, typename IndexType>
Vector<IndexType> ArgSortAlongHeight(const DataType* to_sort,
                                     const IndexType height,
                                     const IndexType width,
                                     const DataType tolerance,
                                     Vector<IndexType>& sorted_ids) {
  // create ids to return
  sorted_ids.resize(height);
  std::iota(sorted_ids.begin(), sorted_ids.end(), 0);

  internal::ArgSortAlongHeight<stable_sort>(to_sort,
                                            height,
                                            width,
                                            tolerance,
                                            sorted_ids);

  return sorted_ids;
}

template<bool stable_sort = false, typename DataType, typename IndexType>
void UniqueIds(const DataType* flat_2d_array,
               const IndexType height,
               const IndexType width,
               const DataType tolerance,
               Vector<IndexType>& sorted_ids,
               Vector<IndexType>& unique_ids,
               IndexType* inverse) {
  // prepare 2 index arrays one for argsort, one for unique
  sorted_ids.resize(height);
  unique_ids.resize(height);
  for (IndexType i{}; i < height; ++i) {
    sorted_ids[i] = unique_ids[i] = i;
  }

  // argsort along the height. (row-wise argsort)
  internal::ArgSortAlongHeight<stable_sort>(flat_2d_array,
                                            height,
                                            width,
                                            tolerance,
                                            sorted_ids);

  auto is_same = [&](const IndexType& i_a, IndexType& i_b) {
    const DataType* a_ptr = &flat_2d_array[sorted_ids[i_a] * width];
    const DataType* a_end = a_ptr + width;

    const DataType* b_ptr = &flat_2d_array[sorted_ids[i_b] * width];

    for (; a_ptr != a_end; ++a_ptr, ++b_ptr) {
      const DataType ab_diff = *a_ptr - *b_ptr;
      if (std::abs(ab_diff) > tolerance) {
        return false;
      }
    }
    return true;
  };

  // erase ids of duplicates
  auto last = std::unique(unique_ids.begin(), unique_ids.end(), is_same);
  unique_ids.erase(last, unique_ids.end());

  // now, inverse
  if (inverse) {
    IndexType inverse_counter{};
    for (IndexType i{}; i < height; ++i) {
      if (!is_same(unique_ids[inverse_counter], i)) {
        ++inverse_counter;
      }
      inverse[sorted_ids[i]] = inverse_counter;
    }
  }
}

// sort and write back inplace
template<typename IndexType>
void SortIdsAndInverse(const IndexType ids_len,
                       IndexType* ids,
                       const IndexType inverse_len,
                       IndexType* inverse) {

  // argsort indices - arange initialize
  Vector<IndexType> argsort_ids(ids_len);
  Vector<IndexType> argsort_argsort_ids;
  if (inverse) {
    argsort_argsort_ids.resize(ids_len);
    for (IndexType i{}; i < ids_len; ++i) {
      argsort_ids[i] = argsort_argsort_ids[i] = i;
    }
  } else {
    std::iota(argsort_ids.begin(), argsort_ids.end(), 0);
  }

  // first argsort to sort ids
  auto compare_ids1 = [&](const IndexType& a, const IndexType& b) {
    return ids[a] < ids[b];
  };

  std::sort(argsort_ids.begin(),
            argsort_ids.end(),
            compare_ids1); // should be unique

  // make aux
  Vector<IndexType> sorted_ids(ids_len);
  for (IndexType i{}; i < ids_len; ++i) {
    sorted_ids[i] = ids[argsort_ids[i]];
  }

  // fill ids
  for (IndexType i{}; i < ids_len; ++i) {
    ids[i] = sorted_ids[i];
  }

  // process inverse
  if (inverse) {
    Vector<IndexType> sorted_inverse(inverse_len);
    // second argsort to place inverse correctly
    auto compare_ids2 = [&](const IndexType& a, const IndexType& b) {
      return argsort_ids[a] < argsort_ids[b];
    };

    std::sort(argsort_argsort_ids.begin(),
              argsort_argsort_ids.end(),
              compare_ids2);
    for (IndexType i{}; i < inverse_len; ++i) {
      sorted_inverse[i] = argsort_argsort_ids[inverse[i]];
    }
    for (IndexType i{}; i < inverse_len; ++i) {
      inverse[i] = sorted_inverse[i];
    }
  }
}

} // namespace funi
