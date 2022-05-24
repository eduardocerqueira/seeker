#date: 2022-05-24T17:14:31Z
#url: https://api.github.com/gists/e697a676d66a9b657ff796ac43dd08c1
#owner: https://api.github.com/users/ticowiko

def generic_join_op(
    left: AbstractDataStructure,
    right: AbstractDataStructure,
    join_cols: List[str],
) -> AbstractDataStructure:
    return left.join(
        right,
        left_on=join_cols,
        right_on=join_cols,
        how="inner",
        l_modifier=lambda x: f"l_{x}",
        r_modifier=lambda x: f"r_{x}",
        modify_on=False,
    )