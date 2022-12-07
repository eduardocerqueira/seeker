#date: 2022-12-07T16:45:58Z
#url: https://api.github.com/gists/497eabe530f68099a6cf51f63ac05b67
#owner: https://api.github.com/users/alexismanin

import numpy as np
import h5py
from h5py import Dataset

FILE="test_data.h5"

def print_dataset(dataset: Dataset):
    '''
    Prints information and values contained in a HDF Dataset.
    Does not work with HDF Groups.
    '''
    print(dataset.name)
    print("Shape: ", dataset.shape)
    print("Chunks: ", dataset.chunks)
    print("Data type: ", dataset.dtype)
    print("Values:")
    print(dataset[::])
    
def write_then_print(dataset_name: str, numpy_values, chunks = None, file_path: str = FILE, write_mode: str = 'a'):
    '''
    Create a new dataset in input file, then flush file content and print written dataset.
    By default, file is opened in 'append' mode. You can change it to either:
     - 'w': write, create if needed, truncate existing content
     - 'w-': write, fail if file already exists
    '''
    with h5py.File(file_path, write_mode) as f:
        dataset = f.create_dataset(dataset_name, shape = numpy_values.shape, dtype = numpy_values.dtype, chunks = chunks)
        dataset[::] = numpy_values[::]
    
    # close and open back file in read mode, to ensure data has been properly flushed
    with h5py.File(file_path, 'r') as f:
        print_dataset(f[dataset_name])
    
    print("\n-------------\n")

    
#
# WRITE ARRAY DATASETS
#

write_then_print("array_1D", np.arange(0, 50) / 10.0, chunks = (10), write_mode = 'w')

# 2D array. Write values of the form: 1jjii where jj is index on axis 0 and ii index on axis 1
array2d = np.ndarray(shape = (20, 20), dtype = 'int32')
for j in range(0, array2d.shape[0]):
    for i in range(0, array2d.shape[1]):
        array2d[j, i] = 10000 + j * 100 + i
write_then_print("array_2D", array2d)


# 3D array. Write values of the form: 1jjii where kk is index on axis 0, jj is index on axis 1 and ii index on axis 2
array3d = np.ndarray(shape = (20, 20, 20), dtype = 'int32')
for k in range(0, array3d.shape[0]):  
    for j in range(0, array3d.shape[1]):
        for i in range(0, array3d.shape[2]):
            array3d[k, j, i] = 1_000_000 + k * 10_000 + j * 100 + i
write_then_print("array_3D", array3d)

#
# WRITE COMPOUND (TABLE) DATASETS
#

# 1D Table:
dtype = np.dtype([('index', 'int8'),
                  ('index_divided_by_ten', 'float32'),
                  ('index_multiplied_by_ten', 'int32'),
                  ('status', 'S2')
                 ])
table = np.ndarray(shape=(50), dtype=dtype)
for i in range(0, table.shape[0]):
    table[i] = (i, i / 10.0, i * 10, 'OK')
write_then_print("table", table, chunks = (10))


# 2D table
dtype2d = np.dtype([ ('j', 'int8'), ('i', 'int8') ])

table2d = np.ndarray(shape = (20, 20), dtype = dtype2d)
for j in range(0, table3d.shape[0]):  
    for i in range(0, table3d.shape[1]):
        table2d[j, i] = (j, i)
write_then_print("table_2D", table2d, chunks = None)

# 3D table
dtype3d = np.dtype([('k', 'int8'),
                    ('j', 'int8'),
                    ('i', 'int8')
                   ])

table3d = np.ndarray(shape = (20, 20, 20), dtype = dtype3d)
for k in range(0, table3d.shape[0]):  
    for j in range(0, table3d.shape[1]):
        for i in range(0, table3d.shape[2]):
            table3d[k, j, i] = (k, j, i)
write_then_print("table_3D", table3d, chunks = (1, 5, 10))