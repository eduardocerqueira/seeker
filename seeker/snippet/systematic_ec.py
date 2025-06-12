#date: 2025-06-12T17:14:59Z
#url: https://api.github.com/gists/a0b03b594b8e01d43d78360f4288caa6
#owner: https://api.github.com/users/davxy

from sage.all import GF, PolynomialRing

class ErasureCoding:
    def __init__(self, message_blocks=2, total_blocks=6):
        """
        Initialize erasure coding with configurable parameters.
        
        Args:
            message_blocks: Number of 16-bit blocks in the message (default: 2)
            total_blocks: Total number of blocks after encoding (default: 6)
        """
        if message_blocks >= total_blocks:
            raise ValueError("message_blocks must be less than total_blocks")
        
        self.message_blocks = message_blocks
        self.total_blocks = total_blocks
               
        # Define GF(2^16) with the specified irreducible polynomial
        # f_16 = x^16 + x^5 + x^3 + x^2 + 1
        R_temp = PolynomialRing(GF(2), 'x')
        x_temp = R_temp.gen()
        modulus = x_temp**16 + x_temp**5 + x_temp**3 + x_temp**2 + 1
        self.F = GF(2**16, name='x', modulus=modulus)
        self.x = self.F.gen()
               
        # Precompute evaluation points in Cantor basis
        self._precompute_evaluation_points()

        print("Initialized erasure coding with:")
        print(f"  Message blocks: {self.message_blocks}")
        print(f"  Total blocks: {self.total_blocks}")
           
    def _int_to_cantor_basis(self, n):
        """
        Convert a 16-bit integer to Cantor basis representation.
        """
        if not (0 <= n <= 65535):
            raise ValueError("Input must be a 16-bit integer (0 to 65535)")
        
        # Define each v_j as a 16-bit integer
        # Bit position i represents α^i (bit 0 = α^0, bit 1 = α^1, etc.)
        cantor_basis = [
            0b0000000000000001,  # v0 = 1
            0b1010110011001010,  # v1 = α^15 + α^13 + α^11 + α^10 + α^7 + α^6 + α^3 + α
            0b0011110000001110,  # v2 = α^13 + α^12 + α^11 + α^10 + α^3 + α^2 + α
            0b0001011000111110,  # v3 = α^12 + α^10 + α^9 + α^5 + α^4 + α^3 + α^2 + α
            0b1100010110000010,  # v4 = α^15 + α^14 + α^10 + α^8 + α^7 + α
            0b1110110100101110,  # v5 = α^15 + α^14 + α^13 + α^11 + α^10 + α^8 + α^5 + α^3 + α^2 + α
            0b1001000101001100,  # v6 = α^15 + α^12 + α^8 + α^6 + α^3 + α^2
            0b0100000000010010,  # v7 = α^14 + α^4 + α
            0b0110110010011000,  # v8 = α^14 + α^13 + α^11 + α^10 + α^7 + α^4 + α^3
            0b0001000011011000,  # v9 = α^12 + α^7 + α^6 + α^4 + α^3
            0b0110101001110010,  # v10 = α^14 + α^13 + α^11 + α^9 + α^6 + α^5 + α^4 + α
            0b1011100100000000,  # v11 = α^15 + α^13 + α^12 + α^11 + α^8
            0b1111110110111000,  # v12 = α^15 + α^14 + α^13 + α^12 + α^11 + α^10 + α^8 + α^7 + α^5 + α^4 + α^3
            0b1111101100110100,  # v13 = α^15 + α^14 + α^13 + α^12 + α^11 + α^9 + α^8 + α^5 + α^4 + α^2
            0b1111111100111000,  # v14 = α^15 + α^14 + α^13 + α^12 + α^11 + α^10 + α^9 + α^8 + α^5 + α^4 + α^3
            0b1001100100011110,  # v15 = α^15 + α^12 + α^11 + α^8 + α^4 + α^3 + α^2 + α
        ]
        
        result = 0
        for bit_position in range(16):
            if (n & (1 << bit_position)) != 0:
                result ^= cantor_basis[bit_position]
        return result
    
    def _cantor_basis_to_int(self, field_element):
        """
        Convert from Cantor basis representation back to original 16-bit integer.
        (Brute force...)
        """
        # Brute force search since it's only 16 bits
        for candidate in range(65536):
            if self._int_to_cantor_basis(candidate) == field_element:
                return candidate
        # Should never happen if field_element is valid
        raise ValueError(f"Could not find original integer for field element {field_element}")
      
    def _precompute_evaluation_points(self):
        """Precompute the evaluation points using Cantor basis"""
        self.evaluation_points = []       
        for i in range(self.total_blocks):
            eval_point = self._word_to_field_element(i)
            self.evaluation_points.append(eval_point)
    
    def _word_to_field_element(self, word):
        """Convert a 16-bit word (integer) to a field element using Cantor basis"""
        if word < 0 or word >= 2**16:
            raise ValueError("Word must be an integer between 0 and 65535")
        # Convert word to Cantor basis representation first
        cantor_rep = self._int_to_cantor_basis(word)       
        # Then convert to field element using from_integer
        return self.F.from_integer(cantor_rep)
    
    def _field_element_to_word(self, element):
        """Convert a field element to a 16-bit word (integer) using Cantor basis"""
        if element == self.F(0):
            return 0
        # Convert field element to integer representation
        cantor_rep = element.to_integer()
        # Convert from Cantor basis back to original integer
        return self._cantor_basis_to_int(cantor_rep)
    
    def encode(self, message_words):
        """
        Encode a message into code words using systematic encoding.
        """
        if len(message_words) != self.message_blocks:
            raise ValueError(f"Message must contain exactly {self.message_blocks} words")

        # Convert message words to field elements
        message_elements = [self._word_to_field_element(word) for word in message_words]
        
        # First k positions contain the original message
        code_words = message_words.copy()
        
        # Create polynomial from message data using Lagrange interpolation
        # We need to find a polynomial that passes through the message points
        R = PolynomialRing(self.F, 'y')
        
        # Prepare points and values for interpolation
        points = []
        values = []
        for i in range(self.message_blocks):
            points.append(self.evaluation_points[i])
            values.append(message_elements[i])
        
        # Perform Lagrange interpolation to get the polynomial
        pairs = list(zip(points, values))
        p = R.lagrange_polynomial(pairs)
        
        # Generate redundancy blocks by evaluating at remaining points
        for i in range(self.message_blocks, self.total_blocks):
            field_elem = p(self.evaluation_points[i])
            code_words.append(self._field_element_to_word(field_elem))
        
        return code_words
    
    def decode(self, received_words, erasure_positions):
        """
        Decode received words with erasures at given positions.
        """
        # Find available positions
        available_positions = [i for i in range(self.total_blocks) if i not in erasure_positions]
        
        min_required = self.message_blocks
        if len(available_positions) < min_required:
            raise ValueError(f"Too many erasures. Need at least {min_required} blocks, have {len(available_positions)}")
        
        # Check if all message positions are available (no decoding needed)
        message_positions = list(range(self.message_blocks))
        if all(pos not in erasure_positions for pos in message_positions):
            return received_words[:self.message_blocks]
        
        # Use available positions for interpolation
        interp_positions = available_positions[:min_required]
        
        # Prepare points and values for interpolation
        points = []
        values = []       
        for pos in interp_positions:
            points.append(self.evaluation_points[pos])
            field_elem = self._word_to_field_element(received_words[pos])
            values.append(field_elem)
        
        # Perform Lagrange interpolation
        R = PolynomialRing(self.F, 'y')
        pairs = list(zip(points, values))
        p = R.lagrange_polynomial(pairs)
        
        # Recover message by evaluating at the first message_blocks evaluation points
        recovered_words = []
        for i in range(self.message_blocks):
            field_elem = p(self.evaluation_points[i])
            recovered_words.append(self._field_element_to_word(field_elem))
        
        return recovered_words


if __name__ == "__main__":   
    print("Example with default parameters (2 message blocks, 6 total blocks)")
    print("="*60)
    
    # Create erasure coding instance with defaults
    ec = ErasureCoding()  # Uses defaults: message_blocks=2, total_blocks=6
    
    # Create a simple message of 2 16-bit words
    message_words = [0x6369, 0x616f]  # Using hex notation for clarity
    print(f"\nOriginal message: {[hex(w) for w in message_words]}")
    
    # Encode the message
    code_words = ec.encode(message_words)
    print(f"\nGenerated {len(code_words)} code words:")
    for i, word in enumerate(code_words):
        print(f"  Position {i}: {word:04x}")
    
    # Test 1: Single erasure
    print("\n" + "-"*60)
    print("Test 1: erasure at position 3")
    erasure_positions = [3]
    recovered_words = ec.decode(code_words, erasure_positions)
    print(f"Recovered message: {[hex(w) for w in recovered_words]}")
    print(f"Recovery successful: {message_words == recovered_words}")
    
    # Test 2: Maximum erasures (3 erasures for this configuration)
    print("\n" + "-"*60)
    print("Test 2: erasures at positions 1, 3, 5")
    erasure_positions = [1, 3, 5]
    recovered_words = ec.decode(code_words, erasure_positions)
    print(f"Recovered message: {[hex(w) for w in recovered_words]}")
    print(f"Recovery successful: {message_words == recovered_words}")
     
    # Test 3: Maximum erasures (4 erasures for this configuration)
    print("\n" + "-"*60)
    print("Test 3: erasures at positions 0, 1, 3, 5)")
    erasure_positions = [0, 1, 4, 5]
    recovered_words = ec.decode(code_words, erasure_positions)
    print(f"Recovered message: {[hex(w) for w in recovered_words]}")
    print(f"Recovery successful: {message_words == recovered_words}")
