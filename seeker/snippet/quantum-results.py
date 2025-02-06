#date: 2025-02-06T17:10:29Z
#url: https://api.github.com/gists/d13714b2c1191e0a68c08b5ce600d706
#owner: https://api.github.com/users/sam-caldwell

from qiskit_ibm_runtime import QiskitRuntimeService

from collections import Counter


def process_ibm_results(results):
    """
    Process IBM Quantum results to determine the most likely value of x*.
    """
    samples = results['results'][0]['data']['c']['samples']

    # Convert hexadecimal results to binary strings
    num_bits = results['results'][0]['data']['c']['num_bits']
    binary_samples = [bin(int(sample, 16))[2:].zfill(num_bits) for sample in samples]

    # Count occurrences of each binary state
    counts = Counter(binary_samples)

    # Determine the most frequently measured state (x*)
    x_star = max(counts, key=counts.get)

    # Display results
    print("Measurement Results:")
    for state, count in counts.items():
        print(f"State {state}: {count} times")

    print(f"\nMost probable x* = {x_star} (binary)")
    print(f"Most probable x* = {int(x_star, 2)} (decimal)")


service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token= "**********"
)

job = service.job('<redacted>')


job_result = job.result()

process_ibm_results(job_result)
