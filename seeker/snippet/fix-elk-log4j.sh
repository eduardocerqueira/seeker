#date: 2021-12-23T16:37:38Z
#url: https://api.github.com/gists/1e2136d7d74553ea10d06a26d785b49b
#owner: https://api.github.com/users/mattweidner

#! /bin/bash

if [[ -z $ELK ]]; then
    ELK="elasticsearch"
fi

if [[ $ELK = "elasticsearch" ]]; then
    LOG4J_PATH="/usr/share/elasticsearch"
elif [[ $ELK = "logstash" ]]; then
    LOG4J_PATH="/usr/share/logstash"
else
    echo "ELK must be either 'elasticsearch' or 'logstash'"
    exit 1
fi

if [ ! -d $LOG4J_PATH ]; then
    echo "$LOG4J_PATH does not exist. Are you sure this server runs ${ELK}?"
    exit 1
fi

if [[ $1 = "--live" ]]; then
    LIVE=true
else
    LIVE=false
fi

if $LIVE; then
    echo "This is a live run"
else
    echo "This is not a live run; no changes will be made."
fi

echo

if $LIVE; then
    TEMP_BACKUP=$(mktemp --suffix=.tar)

    echo "Creating backup of $LOG4J_PATH in $TEMP_BACKUP"
    cmd="sudo tar cf $TEMP_BACKUP $LOG4J_PATH"
    echo "[$cmd]"
    eval $cmd

    if [[ $? -ne 0 ]]; then
        echo "Unable to create backup of $LOG4J_PATH"
        exit 1
    fi
fi

log4j=$(find $LOG4J_PATH 2> /dev/null | grep -E 'log4j(-[a-z0-9]*)*-2\.1[0-5]\.[0-9]\.jar$')

if [[ $? -ne 0 ]]; then
    echo "No files were found in $LOG4J_PATH"
    exit 0
fi

echo "Found the following affected log4j jar files..."

for file in $log4j
do
    echo $file
done

echo

if $LIVE; then
    TEMPFILE=$(mktemp --suffix=.zip)
    TEMPDIR=$(mktemp -d)

    echo "Fetching the updated apache log4j 2.16.0 package as $TEMPFILE"
    echo

    wget -O ${TEMPFILE} https://archive.apache.org/dist/logging/log4j/2.16.0/apache-log4j-2.16.0-bin.zip

    echo "Extracting the apache log4j 2.16.0 package to $TEMPDIR"

    pushd $TEMPDIR
    unzip $TEMPFILE > /dev/null 2>&1
    popd
fi

COUNT=0

for file in $log4j
do
    COUNT=$(expr $COUNT + 1)

    dir=$(dirname $file)
    base=$(basename $file)
    newfile=$(echo $base | sed -E -e 's/2\.1[0-9]\.[0-9]/2.16.0/')

    path_newfile="${TEMPDIR}/apache-log4j-2.16.0-bin/${newfile}"

    if $LIVE; then
        if [ ! -f $path_newfile ]; then
            echo "Unable to find replacement library ${path_newfile}"
            continue
        fi
    fi

    echo "Replacing $file with $path_newfile"

    cmd="sudo cp -f ${TEMPDIR}/apache-log4j-2.16.0-bin/${newfile} ${dir}"
    echo "[$cmd]"

    $LIVE && eval $cmd

    cmd="sudo rm -f $file"
    echo "[$cmd]"

    $LIVE && eval $cmd

    GEM_PATH=$(echo $file | grep -o -E '(/[^/]*)*/gems/[^/]*')

    if [ ! -z $GEM_PATH ]; then
        echo
        echo "Found ruby GEM $GEM_PATH"
        echo

        jar_dir=$(dirname $file)
        new_jar_dir=$(echo $jar_dir | sed -e 's/2\.1[0-5]\.[0-9]/2.16.0/')

        echo "Renaming ${jar_dir} to ${new_jar_dir}"
        cmd="sudo mv '${jar_dir}' '${new_jar_dir}'"
        echo "[$cmd]"

        $LIVE && eval $cmd

        echo

        GEM_PATH="${GEM_PATH}/lib"

        rb_files=$(find $GEM_PATH -maxdepth 2 -iname "*.rb")

        for rb_file in $rb_files
        do
            echo "Fixing ruby dependency file $rb_file"
            cmd="sudo sed -E -i.bak -e '/org.apache.logging.log4j/ s/2\.1[0-5]\.[0-9]/2.16.0/' $rb_file"
            echo "[$cmd]"

            if $LIVE; then
                eval $cmd
                diff -u "${rb_file}.bak" "$rb_file"
            fi
        done
    fi

    echo "-------------------------------"
    echo
done

if $LIVE; then
    echo "Updated $COUNT log4j libraries"
    echo
fi

if $LIVE; then
    echo "Removing temp file $TEMPFILE and temp directory $TEMPDIR"

    rm -rf $TEMPDIR
    rm -f $TEMPFILE

    echo
    echo "--------------------------------------------------------------------------------------------------------"
    echo "A backup of $LOG4J_PATH has been saved in $TEMP_BACKUP if you need to revert."
    echo "You can delete this file if everything looks okay"
    echo "Don't forget to restart your elasticsearch or logstash process now"
    echo "--------------------------------------------------------------------------------------------------------"
    echo
fi
