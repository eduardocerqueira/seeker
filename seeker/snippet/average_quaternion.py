#date: 2023-04-03T17:06:34Z
#url: https://api.github.com/gists/dcc415799ad4e5f6e06a68b8f9e08410
#owner: https://api.github.com/users/cmower

import numpy as np

def quaternion_average(*quaternions):
  """Compute the average quaternion."""
  # Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
  # "Averaging quaternions." Journal of Guidance, Control, and Dynamics, 2007.
  # https://ntrs.nasa.gov/citations/20070017872
  Q = np.concatenate([q.flatten().reshape(-1, 1) for q in quaternions], axis=1)
  evals, evecs = np.linalg.eig(Q @ Q.T)
  return evecs[:, evals.argmax()]