#date: 2021-12-17T17:11:42Z
#url: https://api.github.com/gists/80fe6f324d45d9826881b77106e38665
#owner: https://api.github.com/users/jmlich

/bin/bash

declare -A list=(
    [glacier-browser]='string(//context[name="MainPage"]/message[source="Browser"]/translation)'
    [glacier-calc]='string(//context[name="glacier-calc"]/message[source="Calculator"]/translation)'
    [glacier-camera]='string(//context[name="CameraPage"]/message[source="Camera"]/translation)'
    [glacier-contacts]='string(//context[name="QObject"]/message[source="Contacts"]/translation)'
    [glacier-dialer]='string(//context[name="FirstPage"]/message[source="Dialer"]/translation)'
    [glacier-filemuncher]='string(//context[name="QObject"]/message[source="Files Browser"]/translation)'
    [glacier-gallery]='string(//context[name="QObject"]/message[source="Gallery"]/translation)'
    [glacier-music]='string(//context[name="PlayerPage"]/message[source="Music"]/translation)'
    [glacier-packagemanager]='string(//context[name="MainPage"]/message[source="Package manager"]/translation)'
    [glacier-settings]='string(//context[name="QObject"]/message[source="Settings"]/translation)'
    [glacier-testtool]='string(//context[name="QObject"]/message[source="Hardware test"]/translation)'
    [glacier-weather]='string(//context[name="MainPage"]/message[source="Weather"]/translation)'
)

for repo in ${!list[@]}; do
    query="${list[$repo]}"

    desktop_file=$(find $repo -name "$repo"'.desktop')

    for fn in $(find $repo -name '*.ts' ! -name '*depend*' -type f); do
        lang=$(xmllint --xpath 'string(//TS/@language)' "$fn") #'
        if [ -z "$lang" ]; then
            continue
        fi

        local_name=$(xmllint --xpath "$query" "$fn")
        if [ -z "$local_name" ]; then
            continue
        fi

        if ! grep -q "^Name\[$lang\]=" $desktop_file; then
            echo "Appending $repo / $lang"
            echo "Name[$lang]=$local_name" >> $desktop_file
        fi


    done

done
