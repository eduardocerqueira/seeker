#date: 2021-11-22T17:08:13Z
#url: https://api.github.com/gists/b36c51b65c9e0ac7e8054b8aec3adf41
#owner: https://api.github.com/users/taikedzierski-ldx

#!/usr/bin/env bash

set -euo pipefail

# This is a semi-intelligent API sucker for the Love2D documentation wiki
# It downloads the bare essentials for offline documentation reading without traversing the entire wiki

# We use some specific page because they sublink to locations in the wiki
#  that are of direct use.
base_sources=(
    # Useful to have offline directly
    "https://love2d.org/wiki/Category:Tutorials"
    "https://love2d.org/wiki/Category:Snippets"

    # Useful to be aware of, even if accessing them must be done whilst online
    "https://love2d.org/wiki/Category:Libraries"
    )

# Delay for $interval seconds to prevent DDoS-ing the server
interval=1


main() {
    # This script can be called without arguments to run the full download
    # Or you can specifically execute certain steps (the function names)

    if [[ -z "$*" ]]; then
        # Encapsulate in subshells
        #  to allow running standalone
        (download_wiki)
        (cleanup_pages)
        (adjust_styles)
    else
        for subcommand in "$@"; do
            ("$subcommand")
        done
    fi
}


download_wiki() {
    command=(
        # Ol' buddy :)
        wget

        # Follow links (up til --level)
        --recursive

        # Only recursively fetch once
        # This will get all the main pages we're after
        --level=1

        # If the destination file exists already, we should not re-attempt download,
        #  when using multiple base sources. This is checked via the local file
        #  and server file's timestamps, therefore:
        # Adjust file timestamps to match server timestamps
        --timestamping

        # Convert absolute links to work locally
        --convert-links
        --html-extension

        # Don't track URLs upwards
        --no-parent

        # Don't DDoS the server
        --wait=$interval

        # Don't produce pointless terminal output, progress, etc
        --no-verbose
        )


    if [[ ! -f .wgetrc ]]; then
        echo "robots = off" > ./.wgetrc
    fi

    for wiki_url in "${base_sources[@]}"; do
        "${command[@]}" "$wiki_url"
    done
}


switch_in() {
    cd love2d.org/wiki || exit 1
}


cleanup_pages() {
    switch_in

    # Remove some useless pages, including the internationalised Main pages
    #  (which will not link appropriately any further)
    rm -r Talk:* Special:* Main_Page_*.html

    # Files with colons in them may not resolve correctly
    # Identify those files specifically, and adjust them in source pages
    colonics=(*:*)

    echo "Adjusting colonic filenames in sources ..."
    colopatterns=(:)
    for colonic in "${colonics[@]}"; do
        colopatterns+=(-e "s/$colonic/${colonic//:/__}/g")
    done
    sed "${colopatterns[@]:1}" -i *.html

    echo "... done."

    rename 's/:/__/g' *.html
}


adjust_styles() {
    # Download styles and patch main stylesheet links
    switch_in

    un_minify="true"

    styles_link="https://love2d.org/w/load.php?debug=${un_minify}&lang=en&modules=ext.pygments%7Cmediawiki.legacy.commonPrint%2Cshared%7Cmediawiki.sectionAnchor%7Cmediawiki.skinning.content.externallinks%7Cmediawiki.skinning.interface%7Cskins.love.styles&only=styles&skin=love"

    wget "$styles_link" -O styles.css

    sed 's|rel="stylesheet" href="/w/load.php|rel="stylesheet" href="styles.css|' -i *.html
}


zipit() {
    zipname="love_wiki-$(date '+%F').zip"
    python -m zipfile -c "$zipname" ./love2d.org/ && echo "Created $zipname"
}


main "$@"