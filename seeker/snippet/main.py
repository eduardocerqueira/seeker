#date: 2023-04-24T16:58:10Z
#url: https://api.github.com/gists/686c975913d10fe1b57908f7c778c43e
#owner: https://api.github.com/users/mypy-play



def example_of_partial_type_append() -> None:
    example_list = []
    reveal_type(example_list)
    example_list.append(1)
    reveal_type(example_list)
    # note that their approach is not equivalent to function-level
    # constraint solving, which would infer list[int | str] here
    # but if mypy has to infer they do it based on the first use
    example_list.append("hello")


def filter_on_ints(x: int) -> bool:
    ...
    
def example_of_partial_type_via_call() -> None:
    example_list = []
    reveal_type(example_list)
    _ = filter(filter_on_ints, example_list)
    reveal_type(example_list)
    
    
condition: bool = ...    

def example_of_partial_type_with_join() -> None:
    """
    This example exposes something about how MyPy implements partial types:
    it looks like the very first syntactic use of a partial type will
    effectively back-propagate the type, because the second part of the branch
    here throws a type error.
    
    A naive implementation in Pyre would likely overwrite the partial type in
    our fixpoint context, which means we would wind up allowing this and
    joining the types.
    
    The mypy approach is likely easier for users to understand. It could be
    done in Pyre, e.g. by using a global lookup table, but it wouldn't conform
    to our usual fixpoint logic.
    """
    example_list = []
    if condition:
        example_list.append(1)
        reveal_type(example_list)
    else:
        example_list.append("hello")
        reveal_type(example_list)
    reveal_type(example_list)