#date: 2024-04-04T17:08:53Z
#url: https://api.github.com/gists/d6d2b4aed668d21aaf30e8246d1a4e21
#owner: https://api.github.com/users/k8si

#!/bin/bash
#
# Script requires:
# - python3 (tested with 3.11)
# - wget
#
# How to run:
#  $ git clone git@github.com:Mozilla-Ocho/llamafile.git && cd llamafile
#  $ wget <url of this gist>
#  $ ./test-llamafile-commit-with-minilm.sh <commit hash>
#

COMMIT=$1
MODEL=leliuga_all-MiniLM-L6-v2.F32.gguf

if [ -z "${COMMIT}" ]
then
  echo "USAGE: ${0} <commit hash>"
  exit 1
fi

# get current commit so we can revert back afterwards
ORIG_COMMIT=$(git rev-parse HEAD)

setup() {
  # download cosmocc-3.3.3 if necessary
  if [ ! -d ".cosmocc" ]; then
    ./build/download-cosmocc.sh .cosmocc/3.3.3 3.3.3 e4d0fa63cd79cc3bfff6c2d015f1776db081409907625aea8ad40cefc1996d08
  fi
  MAKE_CMD=.cosmocc/3.3.3/bin/make

  # download the model
  wget -nc https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.F32.gguf -O ${MODEL}

  # setup python venv
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install requests
}

build_from_commit() {
  "${MAKE_CMD}" -j8
  "${MAKE_CMD}" install PREFIX=builds/${COMMIT}
  LLAMAFILE=builds/${COMMIT}/bin/llamafile
}

cleanup() {
  # Post-build cleanup
  # Sometimes a build will change something in the file: llama.cpp/quantize/quantize.1.asc
  # Undo this change, run make clean, then revert to starting commit
  git checkout -- .
  "${MAKE_CMD}" clean
  git checkout ${ORIG_COMMIT}
}

run_test() {
  logfile="${COMMIT}.log"
  "${LLAMAFILE}" --server --nobrowser --embedding --model ${MODEL} > ${logfile} 2>&1 &
  serverpid=$!
  if [ $? -ne 0 ]
  then
    printf "%s: error during server startup\n" "${COMMIT}"
    cleanup
    exit 1
  fi

  sleep 5  # wait for server to finish startup

  # try to call the server's /embedding endpoint
  python -c "import requests; response = requests.post(url='http://localhost:8080/embedding', json={'content': 'Apples are red.'}); response.raise_for_status(); assert sum(response.json()['embedding']) != 0"

  if [ $? -ne 0 ]
  then
    printf "%s: error calling /embedding endpoint\n" "${COMMIT}"
    kill ${serverpid} || printf "server process already died\n"
    cleanup
    exit 1
  else
    kill ${serverpid} || printf "server process already died\n"
  fi
}

setup

git checkout ${COMMIT}
build_from_commit
run_test
printf "\n\nPASSED: %s\n\n" "${COMMIT}" # test passed if we got to this point without exiting
cleanup
