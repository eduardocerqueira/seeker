#date: 2024-02-06T17:09:42Z
#url: https://api.github.com/gists/c60bd3519962e021b694d86598570d77
#owner: https://api.github.com/users/elmimmo

#!/bin/zsh

# Render multiple Blender documents and their scenes from the command line.
# Only scenes named with the suffix "_o" will be rendered.
#
# Author: Jorge Hernández Valiñani


# Check dependencies
: ${blender:="/Applications/Blender.app/Contents/MacOS/Blender"}

if [ ! -x "$blender" ]; then
	echo "‼️ Error: Blender not found at $blender"
	exit 1
fi


# Function to extract list of scene names ending with "_o" + "~~~" + output path
extract_scenes() {
	blender --background \
	        "$1" \
	        --python-expr 'import bpy; print("\n".join(["···" + scene.name + "~~~" + (scene.render.filepath if scene.render.use_file_extension else "No Output Path specified") for scene in bpy.data.scenes if scene.name.endswith("_o")]))' |\
	 grep "···"
}

render_scene() {
	blender --background \
	        --verbose 0 \
	        "$1" \
	        --scene "$2" \
	        --render-output "$3" \
	        --render-anim
}


# Test command-line arguments
if [ -z "$1" ]; then
	read -A \?"Type or drag files files to render here: " inputs
else
	inputs=( "$@" )
fi


# Loop for each document
for input in "$inputs[@]"
do
	# Validate input
	if [ "$input" = "" ]; then
		continue
	elif [ ! -f "$input" ]; then
		echo "‼️ Error: $input does not exist"
		continue
	fi

	echo "📄 Attempting document $input"

	# Extract scene names
	scenes=( "${(@f)$(extract_scenes "$input")}" )
	
	thisdate=$(date +"%Y-%m-%dT%H-%M-%S")

	# Loop for each scene
	for scene_output in "${scenes[@]}"
	do
		scene=${scene_output#···}
		scene=${scene%%~~~*}

		output_prefix=${scene_output##*/}
		
		output_dir=${scene_output##*~~~}
		output_dir=${output_dir%/*}

		if [ -d "${input%/*}${output_dir}" ]; then
			output_dir=${output_dir}_${thisdate}
		fi

		echo "🎬 Rendering scene \"$scene\" to: ${input%/*}${output_dir}/${output_prefix}"
		render_scene "$input" "$scene" "${input%/*}${output_dir}/${output_prefix}"
	done
done

echo -e "\a"
echo "🏁 Finished"
