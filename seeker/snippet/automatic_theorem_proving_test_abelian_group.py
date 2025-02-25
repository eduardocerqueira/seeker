#date: 2025-02-25T17:01:03Z
#url: https://api.github.com/gists/f35c4fdde7faecc8f1dbc03f8cc13dbf
#owner: https://api.github.com/users/hussenmi

def test_forall_single_predicate():
    """Test the ForallRule with a single predicate for defining abelian groups."""
    print("\n=== Testing ForallRule with Single Predicate ===")
    
    # Create a commutes concept
    print("\n--- Creating commutes concept ---")
    commutes = Concept(
        name="commutes",
        description="Tests if two elements commute under group operation",
        symbolic_definition=lambda G, a, b: Equals(
            Apply(G.op, a, b),
            Apply(G.op, b, a)
        ),
        computational_implementation=lambda G, a, b: G.op(a, b) == G.op(b, a),
        example_structure=ExampleStructure(
            concept_type=ConceptType.PREDICATE,
            component_types=(ExampleType.GROUP, ExampleType.GROUPELEMENT, ExampleType.GROUPELEMENT),
            input_arity=3,
        ),
    )
    
    # Create some simple example groups
    # For simplicity, we'll use integers mod n with addition as examples
    
    # Create Z2 (integers mod 2 with addition)
    Z2 = {
        "carrier": {0, 1},
        "op": lambda a, b: (a + b) % 2,
        "identity": 0,
        "inverse": lambda a: (2 - a) % 2
    }
    
    # Create Z3 (integers mod 3 with addition)
    Z3 = {
        "carrier": {0, 1, 2},
        "op": lambda a, b: (a + b) % 3,
        "identity": 0,
        "inverse": lambda a: (3 - a) % 3
    }
    
    # Create S3 (symmetric group on 3 elements)
    # Using cycle notation: () = identity, (12), (13), (23), (123), (132)
    # We'll represent elements as permutations of [0,1,2]
    id_perm = (0, 1, 2)
    perm12 = (1, 0, 2)
    perm13 = (2, 1, 0)
    perm23 = (0, 2, 1)
    perm123 = (1, 2, 0)
    perm132 = (2, 0, 1)
    
    def compose(p, q):
        return (p[q[0]], p[q[1]], p[q[2]])
    
    def inverse(p):
        result = [0, 0, 0]
        for i in range(3):
            result[p[i]] = i
        return tuple(result)
    
    S3 = {
        "carrier": {id_perm, perm12, perm13, perm23, perm123, perm132},
        "op": compose,
        "identity": id_perm,
        "inverse": inverse
    }
    
    # Add examples for commutes
    # Z2 is abelian, all elements commute
    commutes.add_example((Z2, 0, 0))
    commutes.add_example((Z2, 0, 1))
    commutes.add_example((Z2, 1, 0))
    commutes.add_example((Z2, 1, 1))
    
    # Z3 is abelian, all elements commute
    commutes.add_example((Z3, 0, 1))
    commutes.add_example((Z3, 1, 2))
    
    # S3 is non-abelian, not all elements commute
    commutes.add_example((S3, id_perm, perm12))  # Identity commutes with everything
    commutes.add_example((S3, perm12, perm12))   # Element commutes with itself
    commutes.add_nonexample((S3, perm12, perm23))  # These don't commute
    
    # Create a ForallRule instance
    forall_rule = ForallRule()
    
    # Check if the rule can be applied
    print("\n--- Checking if ForallRule can be applied for single predicate ---")
    can_apply = forall_rule.can_apply(
        commutes, 
        indices=[1, 2],  # Quantify over the group elements a and b
    )
    print(f"Can apply ForallRule to single predicate: {can_apply}")
    
    if can_apply:
        # Apply the ForallRule to create is_abelian
        print("\n--- Applying ForallRule to create is_abelian ---")
        is_abelian = forall_rule.apply(
            commutes, 
            indices=[1, 2],  # Quantify over the group elements a and b
        )
        is_abelian.name = "is_abelian"
        
        # Test various groups
        print("\n--- Testing is_abelian ---")
        
        # Z2 should be abelian
        print(f"is_abelian(Z2) = {is_abelian.compute(Z2)}")
        
        # Z3 should be abelian
        print(f"is_abelian(Z3) = {is_abelian.compute(Z3)}")
        
        # S3 should not be abelian
        print(f"is_abelian(S3) = {is_abelian.compute(S3)}")
        
        # Add as example/nonexample
        if is_abelian.compute(Z2):
            is_abelian.add_example((Z2,))
        else:
            is_abelian.add_nonexample((Z2,))
        
        if is_abelian.compute(Z3):
            is_abelian.add_example((Z3,))
        else:
            is_abelian.add_nonexample((Z3,))
        
        if is_abelian.compute(S3):
            is_abelian.add_example((S3,))
        else:
            is_abelian.add_nonexample((S3,))
        
        # Verify the symbolic definition
        print("\n--- Symbolic Definition ---")
        ctx = Context()
        symbolic = is_abelian.to_lean4(Z2, optimize=True, ctx=ctx)
        print(f"Lean4 translation: {symbolic}")
        
        return is_abelian
    else:
        print("Cannot apply ForallRule for single predicate!")
        return None

if __name__ == "__main__":
    test_forall_rule()  # Test the original case
    test_forall_single_predicate()  # Test the new single predicate case