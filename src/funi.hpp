#pragma once

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace funi {

///
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
template<bool stable = true, typename DataType, typename IndexType>
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

      // return false;
    }
    // happy compier
    // same. return false.
    return false;
  };

  // sort
  if constexpr (stable) {
    std::stable_sort(sorted_ids.begin(),
                     sorted_ids.end(),
                     lexicographical_compare);
  } else {
    std::sort(sorted_ids.begin(), sorted_ids.end(), lexicographical_compare);
  }
}

} // namespace internal

template<bool stable = true, typename DataType, typename IndexType>
Vector<IndexType> ArgSortAlongHeight(const DataType* to_sort,
                                     const IndexType height,
                                     const IndexType width,
                                     const DataType tolerance,
                                     Vector<IndexType>& sorted_ids) {
  // create ids to return
  sorted_ids.resize(height);
  std::iota(sorted_ids.begin(), sorted_ids.end(), 0);

  internal::ArgSortAlongHeight(to_sort, height, width, tolerance, sorted_ids);

  return sorted_ids;
}

template<bool stable = true, typename DataType, typename IndexType>
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
  internal::ArgSortAlongHeight<stable>(flat_2d_array,
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

} // namespace funi
