#date: 2024-06-19T17:06:37Z
#url: https://api.github.com/gists/9d40808aafb0607316c98aa146950dcc
#owner: https://api.github.com/users/Estecka

#!/bin/bash

set -eu -o pipefail
IFS=''

if [[ $# -le 0 ]]
then
	cat >&2 <<EOF
Generates a painting datapack for Minecraft 1.21, using an unzipped texturepack as source.
The paintings' asset_id will be derived from their variant ID, and all paintings will be added to the placeable tag.

Animated paintings are not supported; they will be generated with the wrong size.

This must be called from the root of the texturepack, and will generate data in 
the same folder, turning it into a sort of "all-in-one" pack.

The --pack option will generate separate zip files for the datapack and texturepack, named <pack_name>-data.zip and <pack_name>-assets.zip Their pack.mcmeta will be taken from the files pack-data.mcmeta and pack-assets.mcmeta, respectively. The local pack.mcmeta file is overwritten in the process.

Synopsis: $0 [-r=<pack_resolution>] [--pack=<pack_name>]

Examples: $0 -r=16
          $0 --pack=MyPack
          $0 -r=x32 --pack="Cooler Pack"
EOF
	exit 0;
fi


PACK_NAME="";
RESOLUTION="";

while [[ $# -ge 1 ]]
do 
	if [[ $1 =~ ^-r=x?([0-9]+)$ ]]
	then
		RESOLUTION=${BASH_REMATCH[1]};
	elif [[ $1 =~ ^--pack=(.+)$ ]]
	then
		PACK_NAME=${BASH_REMATCH[1]};
	else
		echo >&2 "Invalid option: $1";
		exit 1;
	fi
	shift;
done;

#******************************************************************************#
# Placeable tag                                                                #
#******************************************************************************#

TAG_PATH=./data/minecraft/tags/painting_variant/placeable.json
TAG_DELIM=""

function start_tag(){
	mkdir -p $(dirname $TAG_PATH);
	cat >$TAG_PATH <<EOF
{
	"values": [
EOF

}

function add_tag(){
	echo >>$TAG_PATH -ne "$TAG_DELIM";
	echo >>$TAG_PATH -ne "\t\t\"$NAMESPACE:$NAME\""

	TAG_DELIM=",\n"
}

function end_tag() {
	cat >>$TAG_PATH <<EOF

	]
}
EOF
}


#******************************************************************************#
# Variant File                                                                 #
#******************************************************************************#

function create_variant_file(){
	local file_path="./data/${NAMESPACE}/painting_variant/${NAME}.json";

	mkdir -p $(dirname $file_path)
	cat >$file_path <<EOF
{
	"asset_id": "$NAMESPACE:$NAME",
	"width":  $WIDTH,
	"height": $HEIGHT
}
EOF
}

#******************************************************************************#
# Datagen                                                                      #
#******************************************************************************#

function process_png(){
	local info=$(file "$1");

	if [[ ! $info =~ ([0-9]+)\ x\ ([0-9]+) ]]
	then
		echo >&2 -e "Unable to parse parse dimensions:\n$info"
		return 1;
	fi

	export  NAME=$(basename "${1%.png}");
	export  WIDTH=$(( ${BASH_REMATCH[1]} / $RESOLUTION ));
	export HEIGHT=$(( ${BASH_REMATCH[2]} / $RESOLUTION ));

	echo >&2 "${WIDTH}x${HEIGHT} $NAMESPACE:$NAME";
	create_variant_file;
	add_tag;
}

function datagen(){
	start_tag;

	for d in ./assets/*
	do if [ -d "$d" ]
	then
		export NAMESPACE=$(basename "$d");
		for f in ./assets/$NAMESPACE/textures/painting/*.png
		do process_png $f;
		done;
	fi
	done;

	end_tag
}


#******************************************************************************#
# Packing                                                                      #
#******************************************************************************#

function pack(){
	local datapack="$PACK_NAME-data.zip";
	local texturepack="$PACK_NAME-assets.zip";

	rm -f "$datapack";
	cp ./pack-data.mcmeta ./pack.mcmeta
	zip "$datapack" -r \
		data/ \
		pack.mcmeta \
		pack.png \
		;

	rm -f "$texturepack";
	cp ./pack-assets.mcmeta ./pack.mcmeta
	zip "$texturepack" -r \
		assets/ \
		pack.mcmeta \
		pack.png \
		;
}


#******************************************************************************#
# Main                                                                         #
#******************************************************************************#

if ! [ -z "$RESOLUTION" ]
then
	datagen;
fi;

if ! [ -z "$PACK_NAME" ]
then
	pack;
fi;

