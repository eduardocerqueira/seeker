#date: 2023-03-24T17:05:17Z
#url: https://api.github.com/gists/d25f4281719497c65c1c8ed9acbf6382
#owner: https://api.github.com/users/Mysteryem

import bpy
import array
import numpy as np
from itertools import chain

# Using the active object for demonstration purposes
me = bpy.context.object.data

"""
Iterating through each vertex in Python can get really slow when the number of vertices is large
"""

# ~17.5ms for 24576 vertices
def vertices_iter():
    # (There might be faster ways to do this iteration, but I've not tried optimizing this much
    # since there's no point when looking at the other options below)
    return array.array('f', chain.from_iterable(v.co for v in me.vertices)).tobytes()

"""
The foreach_get/foreach_set functions can speed things up by doing the iteration in C instead
"""

# ~3.4ms for 24576 vertices
def list_foreach_into_array():
    # foreach_get flattens the vectors, so the list is 3 times the length of the number
    # of vertices, since each co vector has 3 components
    cos = [0] * (len(me.vertices) * 3)
    me.vertices.foreach_get("co", cos)
    return array.array('f', cos).tobytes()

"""
The foreach_get/foreach_set functions also support Python objects that are Buffers such as array.array,
or np.ndarray which can avoid iteration entirely, instead doing a memcpy into the Buffer
"""

# ~0.52ms for 24576 vertices
def foreach_into_buffer_array():
    cos = array.array('f', [0.0]) * (len(me.vertices) * 3)
    # Buffer matches the internal C type of the co data so the foreach_getset C code can directly
    # memcpy into the buffer
    me.vertices.foreach_get("co", cos)
    return cos.tobytes()


"""
But what if you wanted to export double precision floats?
"""

# ~3.1ms for 24576 vertices
def foreach_into_buffer_array_64():
    cos = array.array('d', [0.0]) * (len(me.vertices) * 3)
    # Buffer type doesn't match the internal C type of the co data so the foreach_getset C code can't
    # use the Buffer as a Buffer and has to treat it like a regular Python sequence.
    # See how the performance is pretty similar to list_foreach_into_array
    me.vertices.foreach_get("co", cos)
    return cos.tobytes()

"""
With NumPy its simple to cast an array of data into another type
"""

# ~0.55ms for 24576 vertices
def foreach_into_buffer_np_64():
    # np.empty is also a small optimisation since it doesn't initialize the array contents
    cos = np.empty(len(me.vertices) * 3, dtype=np.single)
    me.vertices.foreach_get("co", cos)
    # NumPy has many 'vectorized' operations, such as type casting and arithmetic, that run in
    # optimized C code rather than requiring you to iterate through the data in Python
    cos = cos.astype(np.double)
    return cos.tobytes()


"""
Newer Blender versions have begun changing the internal mesh format, putting certain parts of
mesh data into separate arrays. Accessing the data through the new means in the Python API should
be faster that using the older means.
"""


# ~0.1ms for 24576 vertices
def foreach_into_buffer_np_64_position():
    position_data = me.attributes["position"].data

    cos = np.empty(len(position_data) * 3, dtype=np.single)
    position_data.foreach_get("vector", cos)

    cos = cos.astype(np.double)
    return cos.tobytes()


def do_timing():
    import timeit

    print()
    # Adjust trials based on mesh size
    trials = 100
    t1 = timeit.timeit("vertices_iter()", globals=globals(), number=trials)/trials*1000
    print(f"{t1:f}ms")

    t2 = timeit.timeit("list_foreach_into_array()", globals=globals(), number=trials)/trials*1000
    print(f"{t2:f}ms")

    t3 = timeit.timeit("foreach_into_buffer_array()", globals=globals(), number=trials)/trials*1000
    print(f"{t3:f}ms")

    t4 = timeit.timeit("foreach_into_buffer_array_64()", globals=globals(), number=trials)/trials*1000
    print(f"{t4:f}ms")

    t5 = timeit.timeit("foreach_into_buffer_np_64()", globals=globals(), number=trials)/trials*1000
    print(f"{t5:f}ms")

    t6 = timeit.timeit("foreach_into_buffer_np_64_position()", globals=globals(), number=trials)/trials*1000
    print(f"{t6:f}ms")

#do_timing()