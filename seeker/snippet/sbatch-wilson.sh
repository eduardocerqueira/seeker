#date: 2024-03-20T16:55:51Z
#url: https://api.github.com/gists/d51be70a333b3559c6aa027a81a5e8fd
#owner: https://api.github.com/users/benjaminhwilliams

  #SBATCH --nodes=1
  #SBATCH --cpus-per-task=20
  #SBATCH --time=04:00:00
  #SBATCH --error=job-%J.err
  #SBATCH --output=job-%J.out
  #SBATCH -A i19-1
  #SBATCH -p cs05r