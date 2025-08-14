#date: 2025-08-14T16:59:26Z
#url: https://api.github.com/gists/5fec61fd5fbc7f3a3faaf09467311875
#owner: https://api.github.com/users/davidberard98

# USAGE:
#   Put this in your triton repo directory.
#   1. Update the [BUILD COMMAND]
#   2. Update the [PYTORCH PATH]
#   3. Update the [TEST COMMAND]
#   4. Run the bisect:
#     $ git bisect start
#     $ git checkout [known good commit]
#     $ git bisect good
#     $ git checkout [known bad commit]
#     $ git bisect bad
#     $ git bisect run bash triton_bisect.sh

# TODO: [BUILD COMMAND] update this build command if you don't want to rebuild llvm
make dev-install-llvm

code=$?
if [ $code -ne 0 ]
then
  exit 125
fi

# TODO: [PYTORCH PATH] update this path to point to the correct pytorch directory (or whatever test you're running
pushd ../pytorch

# TODO: [TEST COMMAND] update this with the test you want to run
python test/inductor/test_cooperative_reductions.py -k test_welford_non_power_of_2_rsplit_persistent_True_x_9_r_8000_rsplit_37

code=$?
exit $code