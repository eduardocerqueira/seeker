#date: 2024-05-31T16:49:56Z
#url: https://api.github.com/gists/b737e77a480a70a4755267dd81f82a68
#owner: https://api.github.com/users/aadarshlalchandani

#!/bin/bash

## Make this script executable: `chmod +x setup.sh`
## RUN COMMAND: `./setup.sh`
## ~OR~
## Directly execute this script from the Gist URL
## wget -q -O - https://gist.github.com/aadarshlalchandani/b737e77a480a70a4755267dd81f82a68/raw | bash

virtual_env_name=env
requirements_filename=requirements.txt
main_py_filename=main.py
run_filename=run.sh

## if you run ./setup.sh reset
if [ "$1" = "reset" ]; then
    rm -rf $virtual_env_name
    rm $requirements_filename
    rm $main_py_filename
    rm $run_filename
fi

## create fresh virtual environment
python -m venv $virtual_env_name
. $virtual_env_name/bin/activate

## directory to store execution logs
logs_dirname=logs
if [ ! -e "$logs_dirname" ]; then
    mkdir "$logs_dirname"
fi

if [ ! -f "$requirements_filename" ]; then
    ## create 'requirements.txt'
    REQUIREMENTS_FILE_CONTENT='## general testing and code review libraries in python
coverage
pytest
pylint

## add more dependencies here...'
    cat <<<"$REQUIREMENTS_FILE_CONTENT" >"$requirements_filename"
fi

if [ ! -f "$main_py_filename" ]; then
    MAIN_PY_FILE_CONTENT='## your code here

if __name__ == "__main__":
    ## driver code here
    print("Hello World!")
'
    cat <<<"$MAIN_PY_FILE_CONTENT" >"$main_py_filename"
fi

## update pip tools
update_libs="pip wheel setuptools"
python -m pip install -U -q $update_libs

## install dependencies
pip install -r "$requirements_filename" -q

if [ ! -f "$run_filename" ]; then
    ## create 'run.sh'
    RUN_FILE_CONTENT='#!/bin/bash

## activate the virtual environment
. env/bin/activate

PROGRAM=$1
TEST_ARG=pytest
LINT_ARG=pylint
PROGRAM_ARG=$2
logs_dirname=logs
logs="$logs_dirname"/"$PROGRAM"_logs.log
: >$logs
line="\n\n"
current_date_time=$(date "+%Y-%m-%d %H:%M:%S")

echo "[$current_date_time] PROGRAM: $PROGRAM $PROGRAM_ARG" >>$logs 2>&1
if [ $PROGRAM = $TEST_ARG  ]; then
    coverage run -m $PROGRAM $PROGRAM_ARG >>$logs 2>&1
    coverage html -d pytest_report
elif [ $PROGRAM = $LINT_ARG  ]; then
    $PROGRAM $PROGRAM_ARG >>$logs 2>&1
else
    python -u $PROGRAM.py $PROGRAM_ARG >>$logs 2>&1
fi

echo -en $line >>$logs 2>&1
echo $PROGRAM $PROGRAM_ARG was executed.
echo'
    cat <<<"$RUN_FILE_CONTENT" >"$run_filename"
    chmod +x "$run_filename"

## usage: ./run.sh <program_name_without_extension> <program_arg_1> <program_arg_2>
fi

echo
echo "Setup Complete."
