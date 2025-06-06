#date: 2025-06-06T17:07:52Z
#url: https://api.github.com/gists/007b5eba2df699d7bcacfc5222cd05a9
#owner: https://api.github.com/users/stanleyedward

# if you run into CUDA mismatch errors even after checking nvcc's version and location
#check these paths
unset LD_LIBRARY_PATH #check if its pointing to the wrong /bin/
unset CUDA_HOME #same for this

#if it still doesnt work
# Set CUDA_HOME to your conda environment (may not be necessary)
export CUDA_HOME=$CONDA_PREFIX

pip install --no-build-isolation -e .