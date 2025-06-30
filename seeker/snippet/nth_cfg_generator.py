#date: 2025-06-30T16:42:18Z
#url: https://api.github.com/gists/6c29cec6af030e41dd2648104123b33b
#owner: https://api.github.com/users/MMonzon00

import itertools
from typing import List, Dict

class CFGGenerator:
    def __init__(self, alphabet: List[str] = ['a', 'b', 'c']):
        self.alphabet = alphabet
        
    def generate_nth_grammar(self, n: int) -> Dict[str, List[str]]:
        """
        Generates the n-th context-free grammar.
        Returns a dictionary with the grammar rules.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        
        # We'll use a systematic enumeration approach
        # This generates grammars in a well-defined order
        grammar_index = 0
        
        # Start with simplest grammars and increase complexity
        for num_rules in range(1, 20):  # Number of production rules
            for num_nonterminals in range(1, min(num_rules + 1, 5)):  # Number of non-terminals
                
                # Generate non-terminal symbols
                nonterminals = [chr(ord('S') + i) for i in range(num_nonterminals)]
                
                # Possible symbols on the right-hand side
                rhs_symbols = list(self.alphabet) + nonterminals + ['ε']
                
                # Generate all possible right-hand sides up to length 3
                all_rhs = ['ε']  # Empty production
                for length in range(1, 4):
                    for combo in itertools.product(rhs_symbols[:-1], repeat=length):  # Exclude ε from combinations
                        all_rhs.append(''.join(combo))
                
                # Generate all possible ways to assign rules to non-terminals
                for rule_assignment in itertools.combinations_with_replacement(range(len(nonterminals)), num_rules):
                    for rhs_assignment in itertools.combinations_with_replacement(range(len(all_rhs)), num_rules):
                        grammar_index += 1
                        
                        if grammar_index == n:
                            # Build the grammar
                            grammar = {nt: [] for nt in nonterminals}
                            
                            for i, (nt_idx, rhs_idx) in enumerate(zip(rule_assignment, rhs_assignment)):
                                nt = nonterminals[nt_idx]
                                rhs = all_rhs[rhs_idx]
                                grammar[nt].append(rhs)
                            
                            return grammar
        
        # If we haven't found the n-th grammar, it's too large
        raise ValueError(f"Grammar index {n} is too large for the current implementation")
    
    def format_grammar(self, grammar: Dict[str, List[str]]) -> str:
        """
        Formats a grammar dictionary into a readable string representation.
        """
        lines = []
        for nonterminal, productions in grammar.items():
            if productions:
                productions_str = ' | '.join(productions)
                lines.append(f"{nonterminal} → {productions_str}")
        return '\n'.join(lines)
    
    def generate_sample_strings(self, grammar: Dict[str, List[str]], max_depth: int = 5, max_strings: int = 10) -> List[str]:
        """
        Generates sample strings from the grammar using derivation.
        """
        strings = set()
        start_symbol = 'S'
        
        def derive(current: str, depth: int) -> None:
            if depth > max_depth or len(strings) >= max_strings:
                return
            
            # Check if current string has no non-terminals
            has_nonterminal = any(c in grammar for c in current)
            if not has_nonterminal:
                if current != 'ε':
                    strings.add(current)
                else:
                    strings.add('')  # Empty string
                return
            
            # Find the leftmost non-terminal and replace it
            for i, char in enumerate(current):
                if char in grammar:
                    for production in grammar[char]:
                        if production == 'ε':
                            new_string = current[:i] + current[i+1:]
                        else:
                            new_string = current[:i] + production + current[i+1:]
                        derive(new_string, depth + 1)
                    break
        
        if start_symbol in grammar:
            derive(start_symbol, 0)
        
        return sorted(list(strings))[:max_strings]


def main():
    # Create the generator
    generator = CFGGenerator()
    
    # Get input from user
    try:
        n = int(input("Ingrese el número n para generar la n-ésima gramática: "))
        
        # Generate the n-th grammar
        grammar = generator.generate_nth_grammar(n)
        
        # Display the grammar
        print(f"\nGramática #{n}:")
        print("=" * 40)
        print(generator.format_grammar(grammar))
        
        # Generate some sample strings
        print("\nCadenas de ejemplo generadas por esta gramática:")
        print("-" * 40)
        sample_strings = generator.generate_sample_strings(grammar)
        if sample_strings:
            for s in sample_strings:
                print(f"'{s}'")
        else:
            print("(No se pudieron generar cadenas)")
            
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")


if __name__ == "__main__":
    while 1:
        main()