#date: 2024-12-26T16:31:33Z
#url: https://api.github.com/gists/d1a4196f822879ae284d8763cd9446b3
#owner: https://api.github.com/users/garyo

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
import sys
import psutil
import os
from multiprocessing import Pool, cpu_count
from functools import partial

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

def estimate_memory_needed(n, d, batch_size):
    """Estimate memory needed in GB"""
    vector_memory = n * d * 4 / 1024**3  # vectors array (float32)
    batch_memory = batch_size * n * 4 / 1024**3  # dot product batch (float32)
    return vector_memory + batch_memory

def process_batch(start_idx, end_idx, vectors, percentile):
    """Process a single batch of dot products"""
    batch_dots = np.abs(vectors[start_idx:end_idx] @ vectors.T)
    np.fill_diagonal(batch_dots[:(end_idx-start_idx), start_idx:end_idx], -1)
    
    batch_max = np.max(batch_dots)
    threshold = np.percentile(batch_dots, 100 - percentile)
    top_dots = batch_dots[batch_dots >= threshold]
    
    return batch_max, top_dots.tolist()

try:
    # Parameters
    n = int(1e5)  # number of vectors
    d = 12000      # dimension
    percentile = 0.1  # top percentage to analyze
    batch_size = 10000  # batch size for dot products
    n_workers = cpu_count() - 1  # Leave one core free for system
    
    # Estimate memory requirements
    estimated_memory = estimate_memory_needed(n, d, batch_size)
    available_memory = 2 * psutil.virtual_memory().total / 1024**3  # Allow 2x physical RAM
    
    print(f"Estimated memory needed: {estimated_memory:.1f} GB")
    print(f"Available memory (including potential swap): {available_memory:.1f} GB")
    print(f"Number of CPU workers: {n_workers}")
    
    if estimated_memory > available_memory:
        raise MemoryError(f"This computation requires {estimated_memory:.1f} GB but only {available_memory:.1f} GB available")
    
    # Generate vectors
    print(f"Generating {n} vectors of dimension {d}...")
    start_time = time.time()
    
    vectors = np.random.normal(0, 1, (n, d)).astype(np.float32)
    # Normalize to unit length
    norms = np.linalg.norm(vectors, axis=1)
    vectors /= norms[:, np.newaxis]
    
    print(f"Vector generation took {time.time() - start_time:.2f} seconds")
    print(f"Current memory usage: {get_memory_usage():.2f} GB")
    
    # Process vectors and compute dot products
    max_dot_product = -1
    dot_products = []
    total_batches = (n + batch_size - 1) // batch_size
    
    print("\nProcessing dot products...")
    batch_start_time = time.time()
    
    # Serial processing for now to debug
    for batch_num, i in enumerate(range(0, n, batch_size)):
        end_idx = min(i + batch_size, n)
        batch_max, top_dots = process_batch(i, end_idx, vectors, percentile)
        
        max_dot_product = max(max_dot_product, batch_max)
        dot_products.extend(top_dots)
        
        # Progress update
        progress = (batch_num + 1) / total_batches
        elapsed_time = time.time() - batch_start_time
        estimated_total = elapsed_time / progress if progress > 0 else 0
        remaining_time = max(0, estimated_total - elapsed_time)
        
        print(f"Batch {batch_num + 1}/{total_batches} ({progress*100:.1f}%) - "
              f"Max: {batch_max:.6f} - Memory: {get_memory_usage():.2f} GB - "
              f"Est. remaining: {remaining_time/60:.1f} min", 
              file=sys.stderr)
    
    # Calculate and display results
    dot_products = np.array(dot_products)
    mean_top = np.mean(dot_products)
    std_top = np.std(dot_products)
    min_angle = np.arccos(max_dot_product) * 180/np.pi
    
    print("\nResults:")
    print(f"Maximum dot product found: {max_dot_product:.6f}")
    print(f"Mean of top {percentile}%: {mean_top:.6f}")
    print(f"Standard deviation of top {percentile}%: {std_top:.6f}")
    print(f"Minimum angle between any two vectors: {min_angle:.2f} degrees")
    
    # Theoretical predictions
    sigma = 1/np.sqrt(d)
    num_pairs = n * (n-1) / 2
    # Expected maximum using extreme value theory
    a_n = np.sqrt(2 * np.log(num_pairs))
    b_n = 1/a_n
    expected_max = sigma * (a_n + 0.5772 * b_n)

    print(f"\nTheoretical predictions:")
    print(f"Expected maximum dot product: {expected_max:.6f}")
    print(f"Expected minimum angle: {np.arccos(expected_max) * 180/np.pi:.2f} degrees")
    print(f"Expected standard deviation of dot products: {sigma:.6f}")

    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(dot_products, bins=50, density=True)
    plt.title(f'Distribution of Top {percentile}% Dot Products\n'
             f'for {n} {d}-dimensional Random Unit Vectors')
    plt.xlabel('Absolute Dot Product')
    plt.ylabel('Density')
    
    plt.legend()
    plt.show()

except Exception as e:
    print(f"Program failed with error: {str(e)}")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    raise
