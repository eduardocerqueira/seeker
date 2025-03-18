#date: 2025-03-18T16:49:56Z
#url: https://api.github.com/gists/370ee9e619809d5c88d49a0f86a3f9cf
#owner: https://api.github.com/users/fulcrum-blog

builder = SamBuilder()
alignment = builder.add_single(
    chrom="chr2", start=200, supplementary=is_supplementary
)
assert include_alignment(
    alignment=alignment, include_supplementary=include_supplementary
) == expected
