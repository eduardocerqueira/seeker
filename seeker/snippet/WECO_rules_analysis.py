#date: 2022-05-10T16:51:36Z
#url: https://api.github.com/gists/e0b48ecc7df732cfe5f0c78725316f7c
#owner: https://api.github.com/users/rsalaza4

# Define outcomes data frame
analysis = pd.DataFrame({'R1_lower':R1_lower,
                        'R1_upper':R1_upper,
                        'R2_lower':R2_lower,
                        'R2_upper':R2_upper,
                        'R3_lower':R3_lower,
                        'R3_upper':R3_upper,
                        'R4_lower':R4_lower,
                        'R4_upper':R4_upper})

analysis.index = df_grouped.index
analysis