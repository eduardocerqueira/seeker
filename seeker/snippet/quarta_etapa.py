#date: 2021-08-31T03:03:55Z
#url: https://api.github.com/gists/63cf93cb0951bfa9a5ba0c0f8fb0c7f6
#owner: https://api.github.com/users/gbrfilipe

import os
cwd = os.getcwd()

today = datetime.today().strftime('%Y-%m-%d')
nome_csv = today + " " + "pessoas_de_hoje.csv"

output_path = cwd + "\\" + "output\\"
full_output_path = output_path + nome_csv