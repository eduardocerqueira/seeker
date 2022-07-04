#date: 2022-07-04T17:03:18Z
#url: https://api.github.com/gists/27dbdff6935c2a6eb77b03d2bcd34014
#owner: https://api.github.com/users/gautamkumar200

# Source: https://nurdabolatov.com/parallel-processing-large-file-in-python

import multiprocessing as mp
import time
import os


def process_line(line):
    # Count frequency for every character
    counter = {}
    for letter in line:
        if letter not in counter:
            counter[letter] = 0
        counter[letter] += 1

    # Find the character with the most frequency using `counter`
    most_frequent_letter = None
    max_count = 0
    for key, value in counter.items():
        if value >= max_count:
            max_count = value
            most_frequent_letter = key
    return most_frequent_letter


def serial_read(file_name):
    results = []
    with open(file_name, 'r') as f:
        for line in f:
            results.append(process_line(line))
    return results


def parallel_read(file_name):
    # Maximum number of processes we can run at a time
    cpu_count = mp.cpu_count()

    file_size = os.path.getsize(file_name)
    chunk_size = file_size // cpu_count

    # Arguments for each chunk (eg. [('input.txt', 0, 32), ('input.txt', 32, 64)])
    chunk_args = []
    with open(file_name, 'r') as f:
        def is_start_of_line(position):
            if position == 0:
                return True
            # Check whether the previous character is EOL
            f.seek(position - 1)
            return f.read(1) == '\n'

        def get_next_line_position(position):
            # Read the current line till the end
            f.seek(position)
            f.readline()
            # Return a position after reading the line
            return f.tell()

        chunk_start = 0
        # Iterate over all chunks and construct arguments for `process_chunk`
        while chunk_start < file_size:
            chunk_end = min(file_size, chunk_start + chunk_size)

            # Make sure the chunk ends at the beginning of the next line
            while not is_start_of_line(chunk_end):
                chunk_end -= 1

            # Handle the case when a line is too long to fit the chunk size
            if chunk_start == chunk_end:
                chunk_end = get_next_line_position(chunk_end)

            # Save `process_chunk` arguments
            args = (file_name, chunk_start, chunk_end)
            chunk_args.append(args)

            # Move to the next chunk
            chunk_start = chunk_end

    with mp.Pool(cpu_count) as p:
        # Run chunks in parallel
        chunk_results = p.starmap(process_chunk, chunk_args)

    results = []
    # Combine chunk results into `results`
    for chunk_result in chunk_results:
        for result in chunk_result:
            results.append(result)
    return results


def process_chunk(file_name, chunk_start, chunk_end):
    chunk_results = []
    with open(file_name, 'r') as f:
        # Moving stream position to `chunk_start`
        f.seek(chunk_start)

        # Read and process lines until `chunk_end`
        for line in f:
            chunk_start += len(line)
            if chunk_start > chunk_end:
                break
            chunk_results.append(process_line(line))
    return chunk_results


def measure(func, *args):
    time_start = time.time()
    result = func(*args)
    time_end = time.time()
    print(f'{func.__name__}: {time_end - time_start}')
    return result


if __name__ == '__main__':
    measure(serial_read, 'input.txt')
    measure(parallel_read, 'input.txt')

