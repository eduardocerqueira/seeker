#date: 2025-02-25T17:00:10Z
#url: https://api.github.com/gists/460a43dfed859579ef1c207fed687f9e
#owner: https://api.github.com/users/hussenmi

class ForallRule(ProductionRule):
    """
    Production rule that creates a new concept by imposing a forall quantifier.
    
    Case 1 (two predicates): Given P(x₁,...,xₙ), Q(y₁,...,yₘ), and indices i₁,...,iₘ:
      R(unquantified args) := ∀ xᵢ₁ ∈ D₁, ..., xᵢₘ ∈ Dₘ, P(x₁,...,xₙ) ⇒ Q(xᵢ₁,...,xᵢₘ)
      
    Case 2 (single predicate): Given P(x₁,...,xₙ) and indices i₁,...,iₘ:
      R(unquantified args) := ∀ xᵢ₁ ∈ D₁, ..., xᵢₘ ∈ Dₘ, P(x₁,...,xₙ)
    
    If no domain is specified, default domains are inferred from types.
    """
    
    def __init__(self):
        super().__init__(
            name="forall",
            description=(
                "Creates a new predicate with universal quantification in two modes:\n"
                "1. Two predicates: Given predicates P and Q, creates R where\n"
                "   R(unquantified args) := ∀ quantified args. P(...) ⇒ Q(...)\n"
                "2. Single predicate: Given predicate P, creates R where\n"
                "   R(unquantified args) := ∀ quantified args. P(...)\n"
                "Default domains are inferred from concept types if not provided."
            ),
            type="Concept",
        )
    
    def can_apply(
        self, 
        *inputs: Entity, 
        indices: List[int], 
        domains: Optional[List[Entity]] = None,
    ) -> bool:
        """
        Check if this rule can be applied to the inputs.
        
        Requirements for both cases:
        1. Indices to quantify must be valid
        2. If domains provided, length must match number of indices
        
        Case 1 (two predicates):
        1. Primary concept must have arity n ≥ 2
        2. Secondary concept must have arity m ≥ 1
        3. Number of indices must equal m
        
        Case 2 (single predicate):
        1. Concept must have arity n ≥ 1
        2. At least one index to quantify over
        3. Not all indices can be quantified (must have at least one unquantified)
        """
        if len(inputs) == 1:
            # Case 2: Single predicate
            concept = inputs[0]
            
            # Concept must be a predicate with arity n ≥ 1
            n = concept.examples.example_structure.input_arity
            if n < 1:
                print("❌ Concept must have arity at least 1.")
                return False
            
            # Need at least one index to quantify
            if not indices:
                print("❌ Must specify at least one index to quantify.")
                return False
            
            # Not all indices can be quantified
            if len(indices) >= n:
                print("❌ Cannot quantify over all arguments.")
                return False
            
            # Check each index is valid
            for idx in indices:
                if idx < 0 or idx >= n:
                    print(f"❌ Index {idx} is out of range for concept with arity {n}.")
                    return False
            
            # If domains provided, length must match number of indices
            if domains is not None and len(domains) != len(indices):
                print("❌ Number of domains must equal number of indices.")
                return False
            
            return True
            
        elif len(inputs) == 2:
            # Case 1: Two predicates (original case)
            primary, secondary = inputs
            
            # Primary must be a predicate with arity n ≥ 2
            n = primary.examples.example_structure.input_arity
            if n < 2:
                print("❌ Primary concept must have arity at least 2.")
                return False
            
            # Secondary must be a predicate; let m be its arity
            m = secondary.examples.example_structure.input_arity
            if m < 1:
                print("❌ Secondary concept must have arity at least 1.")
                return False
            
            # Number of indices must equal m
            if len(indices) != m:
                print("❌ The number of indices must equal the arity of the secondary concept.")
                return False
            
            # Check that each index is valid for the primary concept
            for idx in indices:
                if idx < 0 or idx >= n:
                    print(f"❌ Index {idx} is out of range for primary concept with arity {n}.")
                    return False
            
            # If domains provided, length must match number of indices
            if domains is not None and len(domains) != m:
                print("❌ Number of domains must equal the number of indices.")
                return False
            
            return True
        
        else:
            print("❌ ForallRule requires either one or two input concepts.")
            return False
    
    def apply(
        self,
        *inputs: Entity,
        indices: List[int],
        domains: Optional[List[Entity]] = None,
    ) -> Entity:
        """
        Apply the forall rule to create a new concept.
        
        Case 1 (two predicates):
          R(unquantified args) := ∀ xᵢ₁ ∈ D₁, ..., xᵢₘ ∈ Dₘ, P(x₁,...,xₙ) ⇒ Q(xᵢ₁,...,xᵢₘ)
          
        Case 2 (single predicate):
          R(unquantified args) := ∀ xᵢ₁ ∈ D₁, ..., xᵢₘ ∈ Dₘ, P(x₁,...,xₙ)
        """
        if not self.can_apply(*inputs, indices=indices, domains=domains):
            raise ValueError("Cannot apply ForallRule to these inputs")
        
        if len(inputs) == 1:
            # Case 2: Single predicate
            return self._apply_single_predicate(inputs[0], indices, domains)
        else:
            # Case 1: Two predicates
            return self._apply_two_predicates(inputs[0], inputs[1], indices, domains)
    
    def _apply_single_predicate(
        self,
        concept: Entity,
        indices: List[int],
        domains: Optional[List[Entity]] = None,
    ) -> Entity:
        """Apply the forall rule to a single predicate."""
        concept_arity = concept.examples.example_structure.input_arity
        
        # Determine which indices to keep (not quantified)
        kept_indices = [i for i in range(concept_arity) if i not in indices]
        
        # Infer domains if not provided
        if domains is None:
            domains = self._infer_domains(concept, indices)
        
        # Create a new concept with reduced arity
        concept_types = concept.examples.example_structure.component_types
        new_types = tuple(concept_types[i] for i in kept_indices)
        
        new_concept = Concept(
            name=f"forall_{concept.name}",
            description=f"Universal quantification over {concept.name}",
            symbolic_definition=lambda *args: self._build_single_forall_expr(
                concept, args, indices, kept_indices, domains
            ),
            computational_implementation=lambda *args: self._compute_single_forall(
                concept, args, indices, kept_indices
            ),
            example_structure=ExampleStructure(
                concept_type=ConceptType.PREDICATE,
                component_types=new_types,
                input_arity=len(kept_indices),
            ),
        )
        
        # Store metadata for use in transform_examples
        new_concept._indices = indices
        new_concept._kept_indices = kept_indices
        new_concept._is_implication = False
        
        # Transform examples
        self._transform_examples(new_concept, concept, None)
        
        return new_concept
    
    def _apply_two_predicates(
        self,
        primary: Entity,
        secondary: Entity,
        indices: List[int],
        domains: Optional[List[Entity]] = None,
    ) -> Entity:
        """Apply the forall rule to two predicates (P => Q)."""
        primary_arity = primary.examples.example_structure.input_arity
        
        # Determine which indices to keep (not quantified)
        kept_indices = [i for i in range(primary_arity) if i not in indices]
        
        # Infer domains if not provided
        if domains is None:
            domains = self._infer_domains(primary, indices)
        
        # Create a new concept with reduced arity
        primary_types = primary.examples.example_structure.component_types
        new_types = tuple(primary_types[i] for i in kept_indices)
        
        new_concept = Concept(
            name=f"forall_{primary.name}_{secondary.name}",
            description=f"Universal quantification over {primary.name} implying {secondary.name}",
            symbolic_definition=lambda *args: self._build_impl_forall_expr(
                primary, secondary, args, indices, kept_indices, domains
            ),
            computational_implementation=lambda *args: self._compute_impl_forall(
                primary, secondary, args, indices, kept_indices
            ),
            example_structure=ExampleStructure(
                concept_type=ConceptType.PREDICATE,
                component_types=new_types,
                input_arity=len(kept_indices),
            ),
        )
        
        # Store metadata for use in transform_examples
        new_concept._indices = indices
        new_concept._kept_indices = kept_indices
        new_concept._is_implication = True
        
        # Transform examples
        self._transform_examples(new_concept, primary, secondary)
        
        return new_concept
    
    def _infer_domains(self, concept: Concept, indices: List[int]) -> List[Entity]:
        """Infer domains for the given indices based on the concept's types."""
        component_types = concept.examples.example_structure.component_types
        
        domains = []
        for idx in indices:
            example_type = component_types[idx]
            if example_type == ExampleType.NUMERIC:
                domains.append(NatDomain())
            elif example_type == ExampleType.GROUPELEMENT:
                domains.append(GroupElementDomain())
            elif example_type == ExampleType.GROUP:
                domains.append(GroupDomain())
            elif example_type == ExampleType.SET:
                domains.append(SetDomain())
            else:
                # Default to NatDomain if type is unknown
                domains.append(NatDomain())
        
        return domains
    
    def _build_single_forall_expr(
        self,
        concept: Concept,
        args: Tuple[Any, ...],
        indices: List[int],
        kept_indices: List[int],
        domains: List[Entity],
    ) -> Expression:
        """Build symbolic expression with universal quantifiers for a single predicate."""
        # Map the unquantified (kept) arguments to their positions
        arg_map = {}
        for i, kept_idx in enumerate(kept_indices):
            arg_map[kept_idx] = args[i]
        
        # Build the predicate arguments
        concept_args = []
        for i in range(concept.examples.example_structure.input_arity):
            if i in indices:
                # Use PropVar for quantified indices
                concept_args.append(PropVar(f"x{i}"))
            else:
                # Use the actual argument for unquantified indices
                concept_args.append(arg_map[i])
        
        # Create the inner expression: P(...)
        expr = ConceptApplication(concept, *concept_args)
        
        # Wrap with universal quantifiers (starting from the innermost)
        for i, idx in enumerate(reversed(indices)):
            domain_idx = len(indices) - i - 1  # Adjust for reversed order
            expr = Forall(f"x{idx}", domains[domain_idx], expr)
        
        return expr
    
    def _build_impl_forall_expr(
        self,
        primary: Concept,
        secondary: Concept,
        args: Tuple[Any, ...],
        indices: List[int],
        kept_indices: List[int],
        domains: List[Entity],
    ) -> Expression:
        """Build symbolic expression with universal quantifiers and implication."""
        # Map the unquantified (kept) arguments to their positions
        arg_map = {}
        for i, kept_idx in enumerate(kept_indices):
            arg_map[kept_idx] = args[i]
        
        # Build the primary predicate arguments
        primary_args = []
        for i in range(primary.examples.example_structure.input_arity):
            if i in indices:
                # Use PropVar for quantified indices
                primary_args.append(PropVar(f"x{i}"))
            else:
                # Use the actual argument for unquantified indices
                primary_args.append(arg_map[i])
        
        # Build the secondary predicate arguments (all quantified)
        secondary_args = [PropVar(f"x{idx}") for idx in indices]
        
        # Create the implication: P(...) ⇒ Q(...)
        antecedent = ConceptApplication(primary, *primary_args)
        consequent = ConceptApplication(secondary, *secondary_args)
        implication = Implies(antecedent, consequent)
        
        # Wrap with universal quantifiers (starting from the innermost)
        expr = implication
        for i, idx in enumerate(reversed(indices)):
            domain_idx = len(indices) - i - 1  # Adjust for reversed order
            expr = Forall(f"x{idx}", domains[domain_idx], expr)
        
        return expr
    
    def _compute_single_forall(
        self,
        concept: Concept,
        args: Tuple[Any, ...],
        indices: List[int],
        kept_indices: List[int],
    ) -> bool:
        """
        Compute the result of the forall predicate for a single concept.
        This checks if the concept is true for all values in a reasonable range.
        """
        # Map args to their positions in the concept
        concept_args = [None] * concept.examples.example_structure.input_arity
        for i, kept_idx in enumerate(kept_indices):
            concept_args[kept_idx] = args[i]
        
        # For simplicity, check values in a reasonable range
        return self._check_single_forall_in_range(concept, concept_args, indices)
    
    def _compute_impl_forall(
        self,
        primary: Concept,
        secondary: Concept,
        args: Tuple[Any, ...],
        indices: List[int],
        kept_indices: List[int],
    ) -> bool:
        """
        Compute the result of the forall predicate for the implication case.
        This checks if for all values, primary => secondary holds.
        """
        # Map args to their positions in the primary concept
        primary_args = [None] * primary.examples.example_structure.input_arity
        for i, kept_idx in enumerate(kept_indices):
            primary_args[kept_idx] = args[i]
        
        # For simplicity, check values in a reasonable range
        return self._check_impl_forall_in_range(primary, secondary, primary_args, indices)
    
    def _check_single_forall_in_range(
        self,
        concept: Concept,
        concept_args: List[Any],
        indices: List[int],
        max_range: int = 100,
    ) -> bool:
        """
        Check if concept is true for all values of quantified variables.
        """
        # Generate combinations of values for quantified variables
        def check_values(idx_pos=0, current_args=None):
            if current_args is None:
                current_args = concept_args.copy()
            
            if idx_pos >= len(indices):
                # All indices have values, check the concept
                return concept.compute(*current_args)
            
            # Try values for the current index
            current_idx = indices[idx_pos]
            for val in range(max_range):
                current_args[current_idx] = val
                if not check_values(idx_pos + 1, current_args):
                    return False  # Found a counterexample
            
            return True  # No counterexample found in the range
        
        return check_values()
    
    def _check_impl_forall_in_range(
        self,
        primary: Concept,
        secondary: Concept,
        primary_args: List[Any],
        indices: List[int],
        max_range: int = 100,
    ) -> bool:
        """
        Check the forall implication condition for values in a reasonable range.
        """
        # Generate combinations of values for quantified variables
        def check_values(idx_pos=0, current_args=None):
            if current_args is None:
                current_args = primary_args.copy()
            
            if idx_pos >= len(indices):
                # All indices have values, check the implication
                if primary.compute(*current_args):
                    # Primary is true, check if secondary is also true
                    secondary_args = [current_args[idx] for idx in indices]
                    return secondary.compute(*secondary_args)
                return True  # Primary is false, so implication is vacuously true
            
            # Try values for the current index
            current_idx = indices[idx_pos]
            for val in range(max_range):
                current_args[current_idx] = val
                if not check_values(idx_pos + 1, current_args):
                    return False  # Found a counterexample
            
            return True  # No counterexample found in the range
        
        return check_values()
    
    def _transform_examples(
        self, new_concept: Entity, primary: Entity, secondary: Entity = None
    ):
        """
        Transform examples from the primary concept into examples for the forall concept.
        """
        indices = new_concept._indices
        kept_indices = new_concept._kept_indices
        is_implication = getattr(new_concept, "_is_implication", True)
        
        for ex in primary.examples.get_examples():
            if not isinstance(ex.value, tuple):
                continue
            
            # Keep only unquantified indices
            new_value = tuple(
                ex.value[i] for i in kept_indices
            )
            
            try:
                # Compute whether the forall condition holds
                if new_concept.compute(*new_value):
                    new_concept.add_example(new_value)
                else:
                    new_concept.add_nonexample(new_value)
            except Exception as e:
                print(f"Failed to transform example: {str(e)}")