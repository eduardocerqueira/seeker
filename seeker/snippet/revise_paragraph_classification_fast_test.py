#date: 2025-03-04T17:08:20Z
#url: https://api.github.com/gists/e40fb3dd9b0af1f7a5fc73f2ee5236e9
#owner: https://api.github.com/users/prnake

MAX_HEADING_DISTANCE_DEFAULT = 200
GOOD_OR_BAD = {'good', 'bad'}
GOOD_BAD_NEARGOOD = {'good', 'bad', 'neargood'}

def _get_neighbour(i, paragraphs, ignore_neargood, inc, boundary):
    while i + inc != boundary:
        i += inc
        c = paragraphs[i].class_type
        if c in GOOD_OR_BAD:
            return c
        if c == 'neargood' and not ignore_neargood:
            return c
    return 'bad'

def get_prev_neighbour(i, paragraphs, ignore_neargood):
    """
    Return the class of the paragraph at the top end of the short/neargood
    paragraphs block. If ignore_neargood is True, than only 'bad' or 'good'
    can be returned, otherwise 'neargood' can be returned, too.
    """
    return _get_neighbour(i, paragraphs, ignore_neargood, -1, -1)

def get_next_neighbour(i, paragraphs, ignore_neargood):
    """
    Return the class of the paragraph at the bottom end of the short/neargood
    paragraphs block. If ignore_neargood is True, than only 'bad' or 'good'
    can be returned, otherwise 'neargood' can be returned, too.
    """
    return _get_neighbour(i, paragraphs, ignore_neargood, 1, len(paragraphs))

def revise_paragraph_classification(paragraphs, max_heading_distance=MAX_HEADING_DISTANCE_DEFAULT):
    """
    Context-sensitive paragraph classification. Assumes that classify_pragraphs
    has already been called.
    """

    # Attention: This is a fix for the bug in the original code.
    # Copy the context free class to the class_style
    # This handles the headings as described in the
    # documentation
    for paragraph in paragraphs:
        paragraph.class_type = paragraph.cf_class

    # good headings
    for i, paragraph in enumerate(paragraphs):
        if not (paragraph.heading and paragraph.class_type == 'short'):
            continue
        j = i + 1
        distance = 0
        while j < len(paragraphs) and distance <= max_heading_distance:
            if paragraphs[j].class_type == 'good':
                paragraph.class_type = 'neargood'
                break
            distance += len(paragraphs[j].text)
            j += 1

    # classify short
    new_classes = {}
    for i, paragraph in enumerate(paragraphs):
        if paragraph.class_type != 'short':
            continue
        prev_neighbour = get_prev_neighbour(i, paragraphs, ignore_neargood=True)
        next_neighbour = get_next_neighbour(i, paragraphs, ignore_neargood=True)
        if prev_neighbour == 'good' and next_neighbour == 'good':
            new_classes[i] = 'good'
        elif prev_neighbour == 'bad' and next_neighbour == 'bad':
            new_classes[i] = 'bad'
        # it must be set(['good', 'bad'])
        elif (prev_neighbour == 'bad' and get_prev_neighbour(i, paragraphs, ignore_neargood=False) == 'neargood') or \
             (next_neighbour == 'bad' and get_next_neighbour(i, paragraphs, ignore_neargood=False) == 'neargood'):
            new_classes[i] = 'good'
        else:
            new_classes[i] = 'bad'

    for i, c in new_classes.items():
        paragraphs[i].class_type = c

    # revise neargood
    for i, paragraph in enumerate(paragraphs):
        if paragraph.class_type != 'neargood':
            continue
        prev_neighbour = get_prev_neighbour(i, paragraphs, ignore_neargood=True)
        next_neighbour = get_next_neighbour(i, paragraphs, ignore_neargood=True)
        if (prev_neighbour, next_neighbour) == ('bad', 'bad'):
            paragraph.class_type = 'bad'
        else:
            paragraph.class_type = 'good'

    # more good headings
    for i, paragraph in enumerate(paragraphs):
        if not (paragraph.heading and paragraph.class_type == 'bad' and paragraph.cf_class != 'bad'):
            continue
        j = i + 1
        distance = 0
        while j < len(paragraphs) and distance <= max_heading_distance:
            if paragraphs[j].class_type == 'good':
                paragraph.class_type = 'good'
                break
            distance += len(paragraphs[j].text)
            j += 1

def test_revise_paragraph_classification(num_paragraphs=100):
    """
    Test function for revise_paragraph_classification.
    Generates random paragraphs and runs the classification algorithm.
    
    Args:
        num_paragraphs: Number of paragraphs to generate for testing
    
    Returns:
        List of paragraphs after classification
    """
    import random
    import string
    from justext.paragraph import Paragraph
    
    # Create random paragraphs
    paragraphs = []
    for i in range(num_paragraphs):
        # Create paragraph with random properties
        p = Paragraph(type('Path', (), {'dom': '', 'xpath': ''}))
        p.cf_class = random.choice(['good', 'bad', 'short', 'neargood'])
        p.heading = random.choice([True, False])
        p.text_nodes = ["short"]
        paragraphs.append(p)
    
    # Run the classification algorithm
    revise_paragraph_classification_fast(paragraphs)
    
    # Print statistics
    class_counts = {'good': 0, 'bad': 0, 'short': 0, 'neargood': 0}
    for p in paragraphs:
        if p.class_type in class_counts:
            class_counts[p.class_type] += 1
    
    print(f"Classification results for {num_paragraphs} paragraphs:")
    for class_type, count in class_counts.items():
        print(f"  {class_type}: {count} paragraphs ({count/num_paragraphs*100:.1f}%)")
    
    return paragraphs

def revise_paragraph_classification_fast(paragraphs, max_heading_distance=MAX_HEADING_DISTANCE_DEFAULT):
    """
    Optimized context-sensitive paragraph classification. Assumes that classify_pragraphs has already been called.
    Complexity is O(n), avoiding repeated traversals by pre-computing neighbor information.
    """
    n = len(paragraphs)
    
    # Attention: This is a fix for the bug in the original code.
    # Copy the context free class to the class_style
    # This handles the headings as described in the
    # documentation
    for paragraph in paragraphs:
        paragraph.class_type = paragraph.cf_class
    
    # Pre-compute the position of the next good/bad element
    next_good_or_bad = [n] * n  # Default to end of paragraph list
    next_good_or_bad_or_neargood = [n] * n
    
    # Pre-compute the position of the previous good/bad element
    prev_good_or_bad = [-1] * n  # Default to beginning of paragraph list
    prev_good_or_bad_or_neargood = [-1] * n
    
    # Step 1: Process good headings
    # Pre-compute text length for each paragraph
    text_lengths = [len(p.text) for p in paragraphs]
    
    # Pre-compute the position of good paragraphs after each position
    next_good_pos = [n] * n
    for i in range(n-1, -1, -1):
        if paragraphs[i].class_type == 'good':
            next_good_pos[i] = i
        elif i < n-1:
            next_good_pos[i] = next_good_pos[i+1]
    
    for i, paragraph in enumerate(paragraphs):
        if not (paragraph.heading and paragraph.class_type == 'short'):
            continue
        
        # Use pre-computed next_good_pos to quickly find the next good paragraph
        j = i + 1
        if j < n and next_good_pos[j] < n:
            # Calculate distance
            distance = sum(text_lengths[k] for k in range(i+1, next_good_pos[j]))
            if distance <= max_heading_distance:
                paragraph.class_type = 'neargood'
    
    # Fill previous element indices from back to front
    for i in range(n-2, -1, -1):
        # good or bad
        if paragraphs[i+1].class_type in GOOD_OR_BAD:
            prev_good_or_bad[i] = i+1
            prev_good_or_bad_or_neargood[i] = i+1
        else:
            prev_good_or_bad[i] = prev_good_or_bad[i+1]
            
        if paragraphs[i+1].class_type in GOOD_BAD_NEARGOOD:
            prev_good_or_bad_or_neargood[i] = i+1
        else:
            prev_good_or_bad_or_neargood[i] = prev_good_or_bad_or_neargood[i+1]
    
    # Fill next element indices from front to back
    for i in range(1, n):
        # good or bad
        if paragraphs[i-1].class_type in GOOD_OR_BAD:
            next_good_or_bad[i] = i-1
            next_good_or_bad_or_neargood[i] = i-1
        else:
            next_good_or_bad[i] = next_good_or_bad[i-1]
            
        if paragraphs[i-1].class_type in GOOD_BAD_NEARGOOD:
            next_good_or_bad_or_neargood[i] = i-1
        else:
            next_good_or_bad_or_neargood[i] = next_good_or_bad_or_neargood[i-1]

    # Step 2: Classify short paragraphs
    new_classes = {}
    for i, paragraph in enumerate(paragraphs):
        if paragraph.class_type != 'short':
            continue
        
        # Use pre-computed indices to get neighbor types
        prev_idx = next_good_or_bad[i]
        next_idx = prev_good_or_bad[i]
        
        prev_neighbour = paragraphs[prev_idx].class_type if prev_idx >= 0 and prev_idx < n else 'bad'
        next_neighbour = paragraphs[next_idx].class_type if next_idx >= 0 and next_idx < n else 'bad'
        
        if prev_neighbour == 'good' and next_neighbour == 'good':
            new_classes[i] = 'good'
        elif prev_neighbour == 'bad' and next_neighbour == 'bad':
            new_classes[i] = 'bad'
        else:
            # Check if there are neargood neighbors
            prev_neargood_idx = next_good_or_bad_or_neargood[i]
            next_neargood_idx = prev_good_or_bad_or_neargood[i]
            
            prev_with_neargood = paragraphs[prev_neargood_idx].class_type if prev_neargood_idx >= 0 and prev_neargood_idx < n else 'bad'
            next_with_neargood = paragraphs[next_neargood_idx].class_type if next_neargood_idx >= 0 and next_neargood_idx < n else 'bad'
            
            if (prev_neighbour == 'bad' and prev_with_neargood == 'neargood') or \
               (next_neighbour == 'bad' and next_with_neargood == 'neargood'):
                new_classes[i] = 'good'
            else:
                new_classes[i] = 'bad'
    
    # Apply new classifications
    for i, c in new_classes.items():
        paragraphs[i].class_type = c
    
    # Note: Interesting that this is not needed.
    # Update indices again, as some paragraphs have changed
    """
    for i in range(n-2, -1, -1):
        if paragraphs[i+1].class_type in GOOD_OR_BAD:
            prev_good_or_bad[i] = i+1
        else:
            prev_good_or_bad[i] = prev_good_or_bad[i+1]
    
    for i in range(1, n):
        if paragraphs[i-1].class_type in GOOD_OR_BAD:
            next_good_or_bad[i] = i-1
        else:
            next_good_or_bad[i] = next_good_or_bad[i-1]
    """
            

    # Step 3: Revise neargood paragraphs
    for i, paragraph in enumerate(paragraphs):
        if paragraph.class_type != 'neargood':
            continue
        
        prev_idx = next_good_or_bad[i]
        next_idx = prev_good_or_bad[i]
        
        prev_neighbour = paragraphs[prev_idx].class_type if prev_idx >= 0 and prev_idx < n else 'bad'
        next_neighbour = paragraphs[next_idx].class_type if next_idx >= 0 and next_idx < n else 'bad'
        
        if (prev_neighbour, next_neighbour) == ('bad', 'bad'):
            paragraph.class_type = 'bad'
        else:
            paragraph.class_type = 'good'
    
    # Step 4: More good headings
    # Pre-compute the position of the first good paragraph after each position
    next_good_pos = [n] * n
    for i in range(n-1, -1, -1):
        if paragraphs[i].class_type == 'good':
            next_good_pos[i] = i
        elif i < n-1:
            next_good_pos[i] = next_good_pos[i+1]
    
    for i, paragraph in enumerate(paragraphs):
        if not (paragraph.heading and paragraph.class_type == 'bad' and paragraph.cf_class != 'bad'):
            continue

        # Use pre-computed next_good_pos to quickly find the next good paragraph
        j = i + 1
        if j < n and next_good_pos[j] < n:
            # Calculate distance
            distance = sum(text_lengths[k] for k in range(i+1, next_good_pos[j]))
            if distance <= max_heading_distance:
                paragraph.class_type = 'good'

def test_performance_comparison(num_paragraphs=10000, always_short=False, print_progress=False):
    """
    Compare the performance of original and optimized functions.
    
    Args:
        num_paragraphs: Number of paragraphs to use for testing
    """
    import random
    import time
    from justext.paragraph import Paragraph
    
    # Create random paragraphs
    paragraphs1 = []
    paragraphs2 = []
    for i in range(num_paragraphs):
        # Original dataset
        p1 = Paragraph(type('Path', (), {'dom': '', 'xpath': ''}))
        # p1.cf_class = random.choice(['good', 'bad', 'short', 'neargood'])
        p1.cf_class = random.choice(['short']) if always_short else random.choice(['good', 'bad', 'short', 'neargood'])
        p1.heading = random.choice([True, False])
        p1.text_nodes = ["short" * random.randint(1, 100)]
        paragraphs1.append(p1)
        
        # Copy dataset for optimized function
        p2 = Paragraph(type('Path', (), {'dom': '', 'xpath': ''}))
        p2.cf_class = p1.cf_class
        p2.heading = p1.heading
        p2.text_nodes = p1.text_nodes
        paragraphs2.append(p2)
    
    # Test original function
    start_time = time.time()
    revise_paragraph_classification(paragraphs1)
    original_time = time.time() - start_time
    
    # Test optimized function
    start_time = time.time()
    revise_paragraph_classification_fast(paragraphs2)
    optimized_time = time.time() - start_time
    
    # Verify results are consistent
    is_same = True
    for i in range(num_paragraphs):
        if paragraphs1[i].class_type != paragraphs2[i].class_type:
            is_same = False
            break
    
    if print_progress:
        # Output performance comparison
        print(f"Performance comparison for {num_paragraphs} paragraphs:")
        print(f"Original function time: {original_time:.6f} seconds")
        print(f"Optimized function time: {optimized_time:.6f} seconds")
        print(f"Performance improvement: {original_time/optimized_time:.2f}x")
        print(f"Results consistency: {'Pass' if is_same else 'Fail'}")
    
    return is_same, original_time, optimized_time

if __name__ == "__main__":

    import time
    
    for num in [1000, 10000, 20000, 100000]:
        print(f"\nTesting with {num} paragraphs:")
        start_time = time.time()
        test_paragraphs = test_revise_paragraph_classification(num)
        elapsed_time = time.time() - start_time
        print(f"Time elapsed: {elapsed_time:.4f} seconds")

    # Run 1000 tests with 1000 paragraphs each
    print("\n===== Running 1000 tests with 1000 paragraphs each =====")
    start_time = time.time()
    success_count = 0
    total_tests = 1000
    
    for i in range(total_tests):
        is_same, _, _ = test_performance_comparison(1000, always_short=False, print_progress=False)

        assert is_same == True, "Optimized function results differ from original function!"

        if is_same:
            success_count += 1
    
    elapsed_time = time.time() - start_time
    print(f"Success rate: {success_count}/{total_tests} ({success_count/total_tests*100:.2f}%)")

    # Run performance comparison tests
    for size in [1000, 5000, 10000, 20000]:
        print(f"\n===== Test data size: {size} =====")
        is_same, orig_time, opt_time = test_performance_comparison(size, always_short=True, print_progress=True)
        if not is_same:
            print("Warning: Optimized function results differ from original function!")