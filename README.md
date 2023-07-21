# funi
Find UNIque float array rows.
[numpy.unique](https://numpy.org/doc/stable/reference/generated/numpy.unique.html) is an awesome function that alleviates headaches, fast.
Haven't you wished that it'd be applicable for 2D float arrays?
`funi` is here to help!
There are two available methods: `axis` and `lexicographic`.
`axis` will first project each array to an axis to sort, where as
`lexicographic` sorts given array in lexicographical manner.

## Install
```bash
pip install funi
```

## Quick Start
```python
import funi
import numpy as np

# create a random array with duplicating entries
arr = np.random.random((10000, 3))
arr = np.vstack((arr, arr, arr))
np.random.shuffle(arr)

# specify tolerance and if you want your unique ids to be stable sorted.
unique_data, unique_ids, inverse = funi.unique_rows(
    arr,
    tolerance=1e-11,
    sorted_index=True,
    method="axis",
)

# use ids to extract unique_data from the original array
assert np.allclose(unique_data, arr[unique_ids])

# use inverse to map unique_data back to the original array
assert np.allclose(arr, unique_data[inverse])

# sorted_index=True gives you sorted unique_ids and corresponding inverse
assert np.alltrue(np.sort(unique_ids) == unique_ids)
```
