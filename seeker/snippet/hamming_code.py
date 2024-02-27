#date: 2024-02-27T17:08:28Z
#url: https://api.github.com/gists/ad6f722e054fc957599af1cb7fba01d8
#owner: https://api.github.com/users/louis030195

def generateParityBits(data):
    """
    Generates parity bits for the given 4-bit data.
    """
    p1 = data[0] ^ data[1] ^ data[3]
    p2 = data[0] ^ data[2] ^ data[3]
    p3 = data[1] ^ data[2] ^ data[3]
    return p1, p2, p3

def encodeHamming(data):
    """
    Encodes the 4-bit data using Hamming(7,4) code.
    """
    p1, p2, p3 = generateParityBits(data)
    # Encoded data: p1, p2, d1, p3, d2, d3, d4
    return [p1, p2, data[0], p3, data[1], data[2], data[3]]

def decodeHamming(encoded):
    """
    Decodes the 7-bit encoded data and corrects a single-bit error.
    """
    p1, p2, p3 = generateParityBits([encoded[2], encoded[4], encoded[5], encoded[6]])
    # Error checking
    error_pos = p1 * 1 + p2 * 2 + p3 * 4
    if error_pos:
        print(f"Error detected at position: {error_pos}")
        encoded[error_pos-1] = 1 - encoded[error_pos-1]  # Correct the error
    
    # Extract the original data
    return [encoded[2], encoded[4], encoded[5], encoded[6]]

# Example usage
data = [1, 0, 1, 1]  # Original data
print("Original data:", data)
encoded = encodeHamming(data)
print("Encoded data:", encoded)

# Introducing a single bit error for demonstration
encoded[2] = 0 if encoded[2] == 1 else 1  # Flip the third bit
print("Encoded data with error:", encoded)

decoded = decodeHamming(encoded)
print("Decoded data:", decoded)
