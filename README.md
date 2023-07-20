# funi
Find UNIque float array rows.
[numpy.unique](https://numpy.org/doc/stable/reference/generated/numpy.unique.html) is an awesome function that alleviates headaches, fast.
Haven't you wished that it'd be applicable for 2D float arrays?
`funi` is here to help!

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

# specify tolerance and
# tolerance is used to compare entries in column-wise.
# consider it as bounding-box edge length for duplicate identification
# with stable_sort + sorted_index unique_id will start from 0
unique_data, unique_ids, inverse = funi.unique_rows(
    arr,
    tolerance=1e-11,
    return_unique=True,
    return_index=True,
    sorted_index=True,
    return_inverse=True,
    stable_sort=True,
)

# use ids to extract unique_data from the original array
assert np.allclose(unique_data, arr[unique_ids])

# use inverse to map unique_data back to the original array
assert np.allclose(arr, unique_data[inverse])

# sorted_index=True gives you sorted unique_ids and corresponding inverse
assert np.alltrue(np.sort(unique_ids) == unique_ids)
```
