#date: 2022-02-25T17:09:28Z
#url: https://api.github.com/gists/e8ea2cd59c22d3d6ae7e532b18184ea6
#owner: https://api.github.com/users/Enegal

#!/usr/bin/env bash
: ${1?'Please specify an input resource pack in the same directory as the script (e.g. ./converter.sh MyResourcePack.zip)'}

# ensure input pack exists
if ! test -f "${1}"; then
   echo "Input resource pack ${1} is not in this directory"
   echo "Please ensure you have entered the filename correctly"
   exit 1
else
  printf "\e[33m[•]\e[m \e[37mInput file ${1} detected\e[m\n"
fi

printf '\e[1;31m%-6s\e[m\n' "
██████████████████████████████████████████████████████████████████████████████
████████████████████████ # <!> # W A R N I N G # <!> # ███████████████████████
██████████████████████████████████████████████████████████████████████████████
███ This script has been provided as is. If your resource pack does not    ███
███ entirely conform the vanilla resource specification, including but not ███
███ limited to, missing textures, improper parenting, improperly defined   ███
███ predicates, and malformed JSON files, among other problems, there is a ███
███ strong possibility this script will fail. Please remedy any potential  ███
███ resource pack formatting errors before attempting to make use of this  ███
███ converter. You have been warned.                                       ███
██████████████████████████████████████████████████████████████████████████████
██████████████████████████████████████████████████████████████████████████████
██████████████████████████████████████████████████████████████████████████████
"

read -p $'\e[37mTo acknowledge and continue, press enter. To exit, press Ctrl+C.:\e[0m

'

# ensure we have all the required dependencies
if command jq --version 2>/dev/null | grep -q "1.6"; then
    printf "\e[32m[+]\e[m \e[37mDependency jq satisfied\e[m\n"
    echo
else
    echo "Dependency jq-1.6 is not satisfied"
    echo "You must install jq-1.6 before proceeding"
    echo "See https://stedolan.github.io/jq/download/"
    echo "Exiting script..."
    exit 1
fi

if command -v sponge >/dev/null 2>&1 ; then
    printf "\e[32m[+]\e[m \e[37mDependency sponge satisfied\e[m\n"
    echo
else
    echo "Dependency sponge is not satisfied"
    echo "You must install sponge before proceeding"
    echo "See https://joeyh.name/code/moreutils/"
    echo "Exiting script..."
    exit 1
fi

if command -v convert >/dev/null 2>&1 ; then
    printf "\e[32m[+]\e[m \e[37mDependency imagemagick satisfied\e[m\n"
    echo
else
    echo "Dependency imagemagick is not satisfied"
    echo "You must install imagemagick before proceeding"
    echo "See https://imagemagick.org/script/download.php"
    echo "Exiting script..."
    exit 1
fi
printf "\e[32m[+]\e[m \e[37mAll dependencies have been satisfied\e[m\n"
echo

# initial configuration
if [[ ${2} != default ]]
then

  printf "\e[36mThis script will now ask some configuration question. Default values are yellow. Simply press enter to use the defaults.\e[m\n"
  echo

  printf "\e[1m\e[37mIs there an existing bedrock pack in this directory with which you would like the output merged? (e.g. input.mcpack)\e[m \e[33m[null]\e[m\n"
  echo
  read -p $'\e[37mInput pack to merge:\e[0m ' merge_input
  echo

  printf "\e[1m\e[37mWhat is the max width dimension we should allow for an input texture without downscaling?\e[m \e[33m[128]\e[m\n"
  echo
  read -p $'\e[37mMax width dimension:\e[0m ' maximum_width_dimension
  echo

  printf "\e[1m\e[37mWhat material should we use for the attachables?\e[m \e[33m[entity_alphatest]\e[m\n"
  printf "\e[3m\e[37mFor more info, see:\e[m \e[36m[https://wiki.bedrock.dev/documentation/materials]\e[m\n"
  echo
  read -p $'\e[37mAttachable material:\e[0m ' attachable_material
  echo

  printf "\e[1m\e[37mWhat material should we use for the blocks? (e.g. opaque)\e[m \e[33m[alpha_test]\e[m\n"
  printf "\e[3m\e[37mFor more info, see:\e[m \e[36m[https://wiki.bedrock.dev/documentation/block-model-materials]\e[m\n"
  echo
  read -p $'\e[37mBlock material:\e[0m ' block_material
  echo

  printf "\e[1m\e[37mFrom what URL should we download the fallback resource pack? (must be a direct link)\e[m \e[33m[null]\e[m\n"
  printf "\e[3m\e[37mIf left blank, we will use our own sources to download the default assets.\e[m\n"
  printf "\e[3m\e[37mEnsure any specified pack has all required predicate textures.\e[m\n"
  printf "\e[3m\e[37mIf your input pack already contains all required predicate textures, use 'none' to skip fallback asset download.\e[m\n"
  echo
  read -p $'\e[37mFallback pack URL:\e[0m ' fallback_pack
  echo
  echo
  echo

fi

printf "\e[37mGenerating Bedrock 3D resource pack with settings:\e[m\n"
printf "\e[37mInput pack to merge:\e[m \e[36m${merge_input:=null}\e[m\n"
printf "\e[37mMax width dimension:\e[m \e[36m${maximum_width_dimension:=128}\e[m\n"
printf "\e[37mAttachable material:\e[m \e[36m${attachable_material:=entity_alphatest}\e[m\n"
printf "\e[37mBlock material:\e[m \e[36m${block_material:=alpha_test}\e[m\n"
printf "\e[37mFallback pack URL:\e[m \e[36m${fallback_pack:=null}\e[m\n"
echo

# decompress our input pack
printf "\e[33m[•]\e[m \e[37mDecompressing input pack\e[m\n"
echo
unzip -q ${1}
printf "\e[32m[+]\e[m \e[37mInput pack decompressed\e[m\n"
echo

# get the current default textures and merge them with our rp
if [[ ${fallback_pack} != none ]]
then
   printf "\e[33m[•]\e[m \e[37mNow downloading the fallback resource pack:\e[m\n"
   echo
fi

if [[ ${fallback_pack} = null ]]
then
  printf "\e[3m\e[37m"
  wget -nv --show-progress -O default_assets.zip https://api.github.com/repos/InventivetalentDev/minecraft-assets/zipball/1.16.5
  printf "\e[m"
  echo
  printf "\e[32m[+]\e[m \e[37mFallback resources downloaded\e[m\n"
  root_folder=($(unzip -Z -1 default_assets.zip | head -1))
fi

if [[ ${fallback_pack} != null &&  ${fallback_pack} != none ]]
then
  printf "\e[3m\e[37m"
  wget -nv --show-progress -O default_assets.zip "${fallback_pack}"
  printf "\e[m"
  echo
  printf "\e[32m[+]\e[m \e[37mFallback resources downloaded\e[m\n"
  root_folder=""
fi

if [[ ${fallback_pack} != none ]]
then
  mkdir ./defaultassetholding
  unzip -q -d ./defaultassetholding default_assets.zip "${root_folder}assets/minecraft/textures/**/*"
  printf "\e[32m[+]\e[m \e[37mFallback resources decompressed\e[m\n"
  cp -n -r "./defaultassetholding/${root_folder}assets/minecraft/textures"/* './assets/minecraft/textures/'
  printf "\e[32m[+]\e[m \e[37mFallback resources merged with target pack\e[m\n"
  rm -rf defaultassetholding
  rm -f default_assets.zip
  printf "\e[31m[X]\e[m \e[37mExtraneous fallback resources deleted\e[m\n"
  echo
fi

# generate a fallback texture
convert -size 16x16 xc:\#FFFFFF ./assets/minecraft/textures/fallbacktexture.png

# setup our initial config
printf "\e[33m[•]\e[m \e[37mIterating through all vanilla associated model JSONs to generate initial predicate config\e[m\n"
printf "\e[3m\e[37mOn a large pack, this may take some time...\e[m\n"
echo

# check if we have block and item folders
if test -d "./assets/minecraft/models/item"; then confarg1="./assets/minecraft/models/item/*.json"; fi
if test -d "./assets/minecraft/models/block"; then confarg2="./assets/minecraft/models/block/*.json"; fi


jq -n '[inputs | {(input_filename | sub("(.+)/(?<itemname>.*?).json"; .itemname)): .overrides?[]?}] |

def maxdur($input):
({
  "carrot_on_a_stick": 25,
  "golden_axe": 32,
  "golden_hoe": 32,
  "golden_pickaxe": 32,
  "golden_shovel": 32,
  "golden_sword": 32,
  "wooden_axe": 59,
  "wooden_hoe": 59,
  "wooden_pickaxe":59,
  "wooden_shovel":59,
  "wooden_sword": 59,
  "fishing_rod": 64,
  "flint_and_steel": 64,
  "warped_fungus_on_a_stick": 100,
  "sparkler": 100,
  "glow_stick": 100,
  "stone_axe": 131,
  "stone_hoe": 131,
  "stone_pickaxe":131,
  "stone_shovel":131,
  "stone_sword": 131,
  "shears": 238,
  "iron_axe": 250,
  "iron_hoe": 250,
  "iron_pickaxe": 250,
  "iron_shovel": 250,
  "iron_sword": 250,
  "trident": 250,
  "crossbow": 326,
  "shield": 336,
  "bow": 384,
  "elytra": 432,
  "diamond_axe": 1561,
  "diamond_hoe": 1561,
  "diamond_pickaxe": 1561,
  "diamond_shovel": 1561,
  "diamond_sword": 1561,
  "netherite_axe": 2031,
  "netherite_hoe": 2031,
  "netherite_pickaxe": 2031,
  "netherite_shovel": 2031,
  "netherite_sword": 2031
} | .[$input] // 1)
;

def namespace:
if contains(":") then sub("\\:(.+)"; "") else "minecraft" end
;

[.[] | to_entries | map( select((.value.predicate.damage != null) or (.value.predicate.damaged != null)  or (.value.predicate.custom_model_data != null)) |
      (if .value.predicate.damage then (.value.predicate.damage * maxdur(.key) | round) else null end) as $damage
    | (if .value.predicate.damaged == 0 then 1 else null end) as $unbreakable
    | (if .value.predicate.custom_model_data then .value.predicate.custom_model_data else null end) as $custom_model_data |
  {
    "item": .key,
      "nbt": ({
        "Damage": $damage,
        "Unbreakable": $unbreakable,
        "CustomModelData": $custom_model_data
      }),
    "path": ("./assets/" + (.value.model | namespace) + "/models/" + (.value.model | sub("(.*?)\\:"; "")) + ".json")

}) | .[]]
| walk(if type == "object" then with_entries(select(.value != null)) else . end)
| to_entries | map( ((.value.geyserID = "gmdl_\(1+.key)", .value.geometry = ("geometry.geysercmd." + "gmdl_\(1+.key)")) | .value))
| to_entries | map( ((.value.geometry = ("geometry.geysercmd." + "gmdl_\(1+.key)")) | .value))
| INDEX(.geyserID)

' ${confarg1} ${confarg2} | sponge config.json
printf "\e[32m[+]\e[m \e[37mInitial predicate config generated\e[m\n"

# get a bash array of all model json files in our resource pack
printf "\e[33m[•]\e[m \e[37mGenerating an array of all model JSON files to crosscheck with our predicate config\e[m\n"
json_dir=($(find ./assets/**/models -type f -name '*.json'))

# ensure all our reference files in config.json exist, and delete the entry if they do not
printf "\e[31m[X]\e[m \e[37mRemoving config entries that do not have an associated JSON file in the pack\e[m\n"
jq '

def real_file($input):
($ARGS.positional | index($input) // null);

map_values(if real_file(.path) != null then . else empty end)

' config.json --args ${json_dir[@]} | sponge config.json

# get a bash array of all our input models
printf "\e[33m[•]\e[m \e[37mCreating a bash array for remaing models in our predicate config\e[m\n"
model_array=($(jq -r '.[].path' config.json))

# find initial parental information
printf "\e[33m[•]\e[m \e[37mDoing an initial sweep for level 1 parentals\e[m\n"
jq -n '

[def namespace: if contains(":") then sub("\\:(.+)"; "") else "minecraft" end;

inputs | {
  "path": (input_filename),
  "parent": ("./assets/" + (.parent | namespace) + "/models/" + ((.parent? // empty) | sub("(.*?)\\:"; "")) + ".json")
  }
]

' ${model_array[@]} | sponge parents.json

# add initial parental information to config.json
printf "\e[31m[X]\e[m \e[37mRemoving config entries with non-supported parentals\e[m\n"
echo
jq -s '

. as $global |

def intest($input_i): ($global | .[0] | map({(.path): .parent}) | add | .[$input_i]? // null);

def gtest($input_g):
["./assets/minecraft/models/block/block.json", "./assets/minecraft/models/block/cube.json", "./assets/minecraft/models/block/cube_column.json", "./assets/minecraft/models/block/cube_directional.json", "./assets/minecraft/models/block/cube_mirrored.json", "./assets/minecraft/models/block/observer.json", "./assets/minecraft/models/block/orientable_with_bottom.json", "./assets/minecraft/models/block/piston_extended.json", "./assets/minecraft/models/block/redstone_dust_side.json", "./assets/minecraft/models/block/redstone_dust_side_alt.json", "./assets/minecraft/models/block/template_single_face.json", "./assets/minecraft/models/block/thin_block.json", "./assets/minecraft/models/builtin/entity.json", "./assets/minecraft/models/builtin/generated.json", "./assets/minecraft/models/item/bow.json", "./assets/minecraft/models/item/chest.json", "./assets/minecraft/models/item/crossbow.json", "./assets/minecraft/models/item/fishing_rod.json", "./assets/minecraft/models/item/generated.json", "./assets/minecraft/models/item/handheld.json", "./assets/minecraft/models/item/handheld_rod.json", "./assets/minecraft/models/item/template_skull.json"]
| index($input_g) // null;

.[1] | map_values(. + ({"parent": (intest(.path) // null)} | if gtest(.parent) == null then . else empty end))
| walk(if type == "object" then with_entries(select(.value != null)) else . end)

' parents.json config.json | sponge config.json


# create our initial directories for bp & rp
printf "\e[33m[•]\e[m \e[37mGenerating initial directory strucutre for our bedrock packs\e[m\n"
mkdir -p ./target/rp/models/blocks/geysercmd && mkdir -p ./target/rp/textures/blocks/geysercmd && mkdir -p ./target/rp/attachables/geysercmd && mkdir -p ./target/rp/animations/geysercmd && mkdir -p ./target/bp/blocks/geysercmd

# copy over our pack.png if we have one
if test -f "./pack.png"; then
    cp ./pack.png ./target/rp/pack_icon.png && cp ./pack.png ./target/bp/pack_icon.png
fi

# generate uuids for our manifests
uuid1=($(uuidgen))
uuid2=($(uuidgen))
uuid3=($(uuidgen))
uuid4=($(uuidgen))

# get pack description if we have one
pack_desc=($(jq -r '(.pack.description // "Geyser 3D Items Resource Pack")' ./pack.mcmeta))

# generate rp manifest.json
printf "\e[33m[•]\e[m \e[37mGenerating resource pack manifest\e[m\n"
jq -c --arg pack_desc "${pack_desc}" --arg uuid1 "${uuid1}" --arg uuid2 "${uuid2}" -n '
{
    "format_version": 2,
    "header": {
        "description": "Adds 3D items for use with a Geyser proxy",
        "name": $pack_desc,
        "uuid": ($uuid1 | ascii_downcase),
        "version": [1, 0, 0],
        "min_engine_version": [1, 16, 100]
    },
    "modules": [
        {
            "description": "Adds 3D items for use with a Geyser proxy",
            "type": "resources",
            "uuid": ($uuid2 | ascii_downcase),
            "version": [1, 0, 0]
        }
    ]
}
' | sponge ./target/rp/manifest.json

# generate bp manifest.json
printf "\e[33m[•]\e[m \e[37mGenerating behavior pack manifest\e[m\n"
jq -c --arg pack_desc "${pack_desc}" --arg uuid1 "${uuid1}" --arg uuid3 "${uuid3}" --arg uuid4 "${uuid4}" -n '
{
    "format_version": 2,
    "header": {
        "description": "Adds 3D items for use with a Geyser proxy",
        "name": $pack_desc,
        "uuid": ($uuid3 | ascii_downcase),
        "version": [1, 0, 0],
        "min_engine_version": [ 1, 16, 100]
    },
    "modules": [
        {
            "description": "Adds 3D items for use with a Geyser proxy",
            "type": "data",
            "uuid": ($uuid4 | ascii_downcase),
            "version": [1, 0, 0]
        }
    ],
    "dependencies": [
        {
            "uuid": ($uuid1 | ascii_downcase),
            "version": [1, 0, 0]
        }
    ]
}
' | sponge ./target/bp/manifest.json

# generate rp terrain_texture.json
printf "\e[33m[•]\e[m \e[37mGenerating resource pack terrain texture definition\e[m\n"
jq -nc '
{
  "resource_pack_name": "vanilla",
  "texture_name": "atlas.terrain",
  "padding": 8,
  "num_mip_levels": 4,
  "texture_data": {}
}
' | sponge ./target/rp/textures/terrain_texture.json

printf "\e[33m[•]\e[m \e[37mGenerating resource pack disabling animation\e[m\n"
# generate our disabling animation
jq -nc '
{
  "format_version": "1.8.0",
  "animations": {
    "animation.geysercmd.disable": {
      "loop": true,
      "override_previous_animation": true,
      "bones": {
        "geysercmd": {
          "scale": 0
        }
      }
    }
  }
}
' | sponge ./target/rp/animations/geysercmd/animation.geysercmd.disable.json
printf "\e[32m[+]\e[m \e[37mInitial pack setup complete\e[m\n"
echo

jq -r '.[] | select(.parent != null) | [.path, .geyserID, .parent] | @tsv | gsub("\\t";",")' config.json | sponge pa.csv
jq -r '.[] | select(.parent == null) | [.path, .geyserID] | @tsv | gsub("\\t";",")' config.json | sponge np.csv

_start=1
_end="$(jq -r '. | length' config.json)"
cur_pos=0

function ProgressBar {
    let _progress=(${1}*100/${2}*100)/100
    let _done=(${_progress}*6)/10
    let _left=60-$_done
    _fill=$(printf "%${_done}s")
    _empty=$(printf "%${_left}s")
printf "\r\e[37m█\e[m \e[37m${_fill// /█}\e[m\e[37m${_empty// /•}\e[m \e[37m█\e[m \e[33m${_progress}%%\e[m"
echo
}

# deal with non-parental models [select(.parent == null)]
while IFS=, read -r file gid
do
    cur_pos=$((cur_pos+1))
    printf "\e[33m[•]\e[m \e[37mStarting conversion of primary model with GeyserID ${gid}\e[m\n"
    # get texture array
    texture_array=($(jq -r 'def namespace: if contains(":") then sub("\\:(.+)"; "") else "minecraft" end; .textures | to_entries | sort_by(.key) | map({(.key): .value}) | add | map("./assets/" + (. | namespace) + "/textures/" + (. | sub("(.*?)\\:"; "")) + ".png") | .[]' ${file}))
    # crop any mcmeta associated files on the first frame and make sure we have a texture
    tex_array_counter=0
    for i in "${texture_array[@]}"
    do
      if test -f "${i}.mcmeta"; then
          magick ${i} -background none -gravity North -extent "%[fx:h<w?h:w]x%[fx:h<w?h:w]" ${i}
          printf "\e[31m[!]\e[m \e[37mCropped .textures[${tex_array_counter}] for use in ${gid} becuase mcmeta was detected\e[m\n"
          rm "${i}.mcmeta"
      fi
      if ! test -f "${i}"; then
          texture_array[${tex_array_counter}]="./assets/minecraft/textures/fallbacktexture.png"
          printf "\e[31m[!]\e[m \e[37mFallback texture was used for ${gid}.textures[${tex_array_counter}]\e[m\n"
      fi
      tex_array_counter=$((tex_array_counter+1))
    done
    tile_dim=($(jq -r '.textures | length | sqrt | ceil' ${file}))
    # find widest image & cap texture_width @ 128
    texture_width=($(identify -format "%w\n" ${texture_array[@]} | sort -n -r -k 1 | head -n 1))
    texture_width=$(($texture_width <= ${maximum_width_dimension} ? $texture_width : ${maximum_width_dimension}))
    # generate stitched texture with imagemagick (cap size @512) (perhaps we can filter to used textures only)
    montage ${texture_array[@]} -background transparent -geometry x${texture_width}+0+0 -tile ${tile_dim}x${tile_dim} -interpolate Integer -filter point ./target/rp/textures/blocks/geysercmd/${gid}.png
    # ensure we do not already have this texture, and reuse in the case that we do
    texture_id=${gid}
    texture_hash=($(md5 -q "./target/rp/textures/blocks/geysercmd/${gid}.png" 2>/dev/null))
    match_count=($(md5 -r ./target/rp/textures/blocks/geysercmd/*.png 2>/dev/null | grep "${texture_hash}" | wc -l))
    if [ ${match_count} -gt 1 ]
    then
        # remove our newly generated duplicate texture and change our texture_id to the match
        rm "./target/rp/textures/blocks/geysercmd/${gid}.png"
        texture_id=($(md5 -r ./target/rp/textures/blocks/geysercmd/*.png | grep "${texture_hash}" | cut -d "." -f 2 | cut -c38-))
        printf "\e[33m[•]\e[m \e[37mReusing texture from ${texture_id} for ${gid}\e[m\n"
    fi

    # append our texture information to rp terrain_texture.json
    jq -c --arg texture_id "${texture_id}" --arg geyser_id "${gid}" '
    .texture_data += {
      ($geyser_id): {
        "textures": ("textures/blocks/geysercmd/" + $texture_id)
      }
    }
    ' ./target/rp/textures/terrain_texture.json | sponge ./target/rp/textures/terrain_texture.json

    # convert our rp geometry file from java to bedrock
    jq --arg binding "c.item_slot == 'head' ? 'head' : q.item_slot_to_bone_name(c.item_slot)" --arg model_name "${gid}" -c '

    def element_array:
    (.textures | to_entries | sort_by(.key) | map({(.key): .value}) | add | keys_unsorted) as $texture_array
    | ($texture_array | length) as $frames
    | (($frames | sqrt) | ceil) as $sides
    | (.texture_size[1] // 16) as $t1
    | .elements | map({
      "origin": [(-.to[0] + 8), (.from[1]), (.from[2] - 8)],
      "size": [.to[0] - .from[0], .to[1] - .from[1], .to[2] - .from[2]],
      "rotation": (if (.rotation.axis) == "x" then [(.rotation.angle | tonumber * -1), 0, 0] elif (.rotation.axis) == "y" then [0, (.rotation.angle | tonumber * -1), 0] elif (.rotation.axis) == "z" then [0, 0, (.rotation.angle | tonumber)] else null end),
      "pivot": (if .rotation.origin then [(- .rotation.origin[0] + 8), .rotation.origin[1], (.rotation.origin[2] - 8)] else null end),
      "uv": (
        def uv_calc($input):
          (if (.faces | .[$input]) then
          (.faces | .[$input].texture[1:] as $input_n | $texture_array | (index($input_n) // index("particle"))) as $pos_n
          | ((.faces | .[$input].uv[0] / $sides) + ((fmod($pos_n; $sides)) * (16 / $sides))) as $fn0
          | ((.faces | .[$input].uv[1] / $sides) + ((($pos_n / $sides) | floor) * (16 / $sides))) as $fn1
          | ((.faces | .[$input].uv[2] / $sides) + ((fmod($pos_n; $sides)) * (16 / $sides))) as $fn2
          | ((.faces | .[$input].uv[3] / $sides) + ((($pos_n / $sides) | floor) * (16 / $sides))) as $fn3 |
          {
            "uv": [($fn0), ($fn1)],
            "uv_size": [($fn2 - $fn0), ($fn3 - $fn1)]
          } else null end);
        {
        "north": uv_calc("north"),
        "south": uv_calc("south"),
        "east": uv_calc("east"),
        "west": uv_calc("west"),
        "up": uv_calc("up"),
        "down": uv_calc("down")
        })
    }) | walk( if type == "object" then with_entries(select(.value != null)) else . end)
    ;

    def pivot_groups:
    (element_array) as $element_array |
    [[.elements[].rotation] | unique | .[] | select (.!=null)]
    | map((
    [(- .origin[0] + 8), .origin[1], (.origin[2] - 8)] as $i_piv |
    (if (.axis) == "x" then [(.angle | tonumber * -1), 0, 0] elif (.axis) == "y" then [0, (.angle | tonumber * -1), 0] else [0, 0, (.angle | tonumber)] end) as $i_rot |
    {
      "parent": "geysercmd_z",
      "pivot": ($i_piv),
      "rotation": ($i_rot),
      "mirror": true,
      "cubes": [($element_array | .[] | select(.rotation == $i_rot and .pivot == $i_piv))]
    }))
    ;

    {
      "format_version": "1.16.0",
      "minecraft:geometry": [{
        "description": {
          "identifier": ("geometry.geysercmd." + ($model_name)),
          "texture_width": 16,
          "texture_height": 16,
          "visible_bounds_width": 4,
          "visible_bounds_height": 4.5,
          "visible_bounds_offset": [0, 0.75, 0]
        },
        "bones": ([{
          "name": "geysercmd",
          "binding": $binding,
          "pivot": [0, 8, 0]
        }, {
          "name": "geysercmd_x",
          "parent": "geysercmd",
          "pivot": [0, 8, 0]
        }, {
          "name": "geysercmd_y",
          "parent": "geysercmd_x",
          "pivot": [0, 8, 0]
        }, {
          "name": "geysercmd_z",
          "parent": "geysercmd_y",
          "pivot": [0, 8, 0],
          "cubes": [(element_array | .[] | select(.rotation == null))]
        }] + (pivot_groups | map(del(.cubes[].rotation)) | to_entries | map( (.value.name = "rot_\(1+.key)" ) | .value)))
      }]
    }

    ' ${file} | sponge ./target/rp/models/blocks/geysercmd/${gid}.json

    # generate our rp animations via display settings
    jq -c --arg model_name "${gid}" '

    {
    	"format_version": "1.8.0",
    	"animations": {
    		("animation.geysercmd." + ($model_name) + ".thirdperson_main_hand"): {
    			"loop": true,
    			"bones": {
    				"geysercmd_x": (if .display.thirdperson_righthand then {
    					"rotation": (if .display.thirdperson_righthand.rotation then [(- .display.thirdperson_righthand.rotation[0]), 0, 0] else null end),
    					"position": (if .display.thirdperson_righthand.translation then [(- .display.thirdperson_righthand.translation[0]), (.display.thirdperson_righthand.translation[1]), (.display.thirdperson_righthand.translation[2])] else null end),
              "scale": (if .display.thirdperson_righthand.scale then [(.display.thirdperson_righthand.scale[0]), (.display.thirdperson_righthand.scale[1]), (.display.thirdperson_righthand.scale[2])] else null end)
    				} else null end),
            "geysercmd_y": (if .display.thirdperson_righthand.rotation then {
    					 "rotation": (if .display.thirdperson_righthand.rotation then [0, (- .display.thirdperson_righthand.rotation[1]), 0] else null end)
    				} else null end),
            "geysercmd_z": (if .display.thirdperson_righthand.rotation then {
    				   "rotation": [0, 0, (.display.thirdperson_righthand.rotation[2])]
    				} else null end),
            "geysercmd": {
               "rotation": [90, 0, 0],
               "position": [0, 13, -3]
            }
    			}
    		},
        ("animation.geysercmd." + ($model_name) + ".thirdperson_off_hand"): {
    			"loop": true,
    			"bones": {
    				"geysercmd_x": (if .display.thirdperson_lefthand then {
    					"rotation": (if .display.thirdperson_lefthand.rotation then [(- .display.thirdperson_lefthand.rotation[0]), 0, 0] else null end),
    					"position": (if .display.thirdperson_lefthand.translation then [(- .display.thirdperson_lefthand.translation[0]), (.display.thirdperson_lefthand.translation[1]), (.display.thirdperson_lefthand.translation[2])] else null end),
              "scale": (if .display.thirdperson_lefthand.scale then [(.display.thirdperson_lefthand.scale[0]), (.display.thirdperson_lefthand.scale[1]), (.display.thirdperson_lefthand.scale[2])] else null end)
    				} else null end),
            "geysercmd_y": (if .display.thirdperson_lefthand.rotation then {
    					 "rotation": (if .display.thirdperson_lefthand.rotation then [0, (- .display.thirdperson_lefthand.rotation[1]), 0] else null end)
    				} else null end),
            "geysercmd_z": (if .display.thirdperson_lefthand.rotation then {
    				   "rotation": [0, 0, (.display.thirdperson_lefthand.rotation[2])]
    				} else null end),
            "geysercmd": {
               "rotation": [90, 0, 0],
               "position": [0, 13, -3]
            }
    			}
    		},
        ("animation.geysercmd." + ($model_name) + ".head"): {
    			"loop": true,
    			"bones": {
            "geysercmd_x": {
    					"rotation": (if .display.head.rotation then [(- .display.head.rotation[0]), 0, 0] else null end),
    					"position": (if .display.head.translation then [(- .display.head.translation[0] * 0.625), (.display.head.translation[1] * 0.625), (.display.head.translation[2] * 0.625)] else null end),
              "scale": (if .display.head.scale then (.display.head.scale | map(. * 0.625)) else 0.625 end)
    				},
            "geysercmd_y": (if .display.head.rotation then {
    					"rotation": [0, (- .display.head.rotation[1]), 0]
    				} else null end),
            "geysercmd_z": (if .display.head.rotation then {
    					"rotation": [0, 0, (.display.head.rotation[2])]
    				} else null end),
            "geysercmd": {
               "position": [0, 19.5, 0]
            }
    			}
    		},
        ("animation.geysercmd." + ($model_name) + ".firstperson_main_hand"): {
    			"loop": true,
    			"bones": {
            "geysercmd": {
    					"rotation": [90, 60, -40],
    					"position": [4, 10, 4],
              "scale": 1.5
    				},
    				"geysercmd_x": {
    					"position": (if .display.firstperson_righthand.translation then [(- .display.firstperson_righthand.translation[0]), (.display.firstperson_righthand.translation[1]), (- .display.firstperson_righthand.translation[2])] else null end),
    					"rotation": (if .display.firstperson_righthand.rotation then [(- .display.firstperson_righthand.rotation[0]), 0, 0] else [0.1, 0.1, 0.1] end),
              "scale": (if .display.firstperson_righthand.scale then (.display.firstperson_righthand.scale) else null end)
    				},
            "geysercmd_y": (if .display.firstperson_righthand.rotation then {
    					"rotation": [0, (- .display.firstperson_righthand.rotation[1]), 0]
    				} else null end),
            "geysercmd_z": (if .display.firstperson_righthand.rotation then {
    					"rotation": [0, 0, (.display.firstperson_righthand.rotation[2])]
    				} else null end)
    			}
    		},
    		("animation.geysercmd." + ($model_name) + ".firstperson_off_hand"): {
    			"loop": true,
    			"bones": {
            "geysercmd": {
    					"rotation": [90, 60, -40],
    					"position": [4, 10, 4],
              "scale": 1.5
    				},
    				"geysercmd_x": {
    					"position": (if .display.firstperson_lefthand.translation then [(.display.firstperson_lefthand.translation[0]), (.display.firstperson_lefthand.translation[1]), (- .display.firstperson_lefthand.translation[2])] else null end),
    					"rotation": (if .display.firstperson_lefthand.rotation then [(- .display.firstperson_lefthand.rotation[0]), 0, 0] else [0.1, 0.1, 0.1] end),
              "scale": (if .display.firstperson_lefthand.scale then (.display.firstperson_lefthand.scale) else null end)
    				},
            "geysercmd_y": (if .display.firstperson_lefthand.rotation then {
    					"rotation": [0, (- .display.firstperson_lefthand.rotation[1]), 0]
    				} else null end),
            "geysercmd_z": (if .display.firstperson_lefthand.rotation then {
    					"rotation": [0, 0, (.display.firstperson_lefthand.rotation[2])]
    				} else null end)
    			}
    		}
    	}
    } | walk( if type == "object" then with_entries(select(.value != null)) else . end)

    ' ${file} | sponge ./target/rp/animations/geysercmd/animation.${gid}.json

    # generate our rp attachable definition
    jq -c -n --arg attachable_material "${attachable_material}" --arg v_main "v.main_hand = c.item_slot == 'main_hand';" --arg v_off "v.off_hand = c.item_slot == 'off_hand';" --arg v_head "v.head = c.item_slot == 'head';" --arg model_name "${gid}" --arg texture_name "${texture_id}" '

    {
      "format_version": "1.10.0",
      "minecraft:attachable": {
        "description": {
          "identifier": ("geysercmd:" + $model_name),
          "materials": {
            "default": $attachable_material,
            "enchanted": $attachable_material
          },
          "textures": {
            "default": ("textures/blocks/geysercmd/" + $texture_name),
            "enchanted": "textures/misc/enchanted_item_glint"
          },
          "geometry": {
            "default": ("geometry.geysercmd." + $model_name)
          },
          "scripts": {
            "pre_animation": [$v_main, $v_off, $v_head],
            "animate": [
              {"thirdperson_main_hand": "v.main_hand && !c.is_first_person"},
              {"thirdperson_off_hand": "v.off_hand && !c.is_first_person"},
              {"thirdperson_head": "v.head && !c.is_first_person"},
              {"firstperson_main_hand": "v.main_hand && c.is_first_person"},
              {"firstperson_off_hand": "v.off_hand && c.is_first_person"},
              {"firstperson_head": "c.is_first_person && v.head"}
            ]
          },
          "animations": {
            "thirdperson_main_hand": ("animation.geysercmd." + $model_name + ".thirdperson_main_hand"),
            "thirdperson_off_hand": ("animation.geysercmd." + $model_name + ".thirdperson_off_hand"),
            "thirdperson_head": ("animation.geysercmd." + $model_name + ".head"),
            "firstperson_main_hand": ("animation.geysercmd." + $model_name + ".firstperson_main_hand"),
            "firstperson_off_hand": ("animation.geysercmd." + $model_name + ".firstperson_off_hand"),
            "firstperson_head": "animation.geysercmd.disable"
          },
          "render_controllers": [ "controller.render.item_default" ]
        }
      }
    }

    ' | sponge ./target/rp/attachables/geysercmd/${gid}.attachable.json

    # generate our bp block definition
    jq -c -n --arg block_material "${block_material}" --arg geyser_id "${gid}" '

    {
        "format_version": "1.16.200",
        "minecraft:block": {
            "description": {
                "identifier": ("geysercmd:" + $geyser_id)
            },
            "components": {
                "minecraft:material_instances": {
                    "*": {
                        "texture": $geyser_id,
                        "render_method": $block_material,
                        "face_dimming": false,
                        "ambient_occlusion": false
                    }
                },
                "tag:geysercmd:example_block": {},
                "minecraft:geometry": ("geometry.geysercmd." + $geyser_id),
                "minecraft:placement_filter": {
                  "conditions": [
                      {
                          "allowed_faces": [
                          ],
                          "block_filter": [
                          ]
                      }
                  ]
                }
            }
        }
    }

    ' | sponge ./target/bp/blocks/geysercmd/${gid}.json

    printf "\e[32m[+]\e[m \e[37m${gid} converted\e[m\n"
    ProgressBar ${cur_pos} ${_end}
    echo
done < np.csv

printf "\e[32m[+]\e[m \e[37mFinished conversion of primary models\e[m\n"
echo
printf "\e[33m[•]\e[m \e[37mStarting conversion of child models\e[m\n"
echo

# deal with parental models [select(.parent != null)]
while IFS=, read -r file gid parental
do
    cur_pos=$((cur_pos+1))
    elements="$(jq -rc '.elements' ${file})"
    element_parent=${file}
    textures="$(jq -rc '.textures' ${file})"
    display="$(jq -rc '.display' ${file})"
    printf "\e[33m[•]\e[m \e[37mStarting conversion attempt for child model with GeyserID ${gid}\e[m\n"

    until [[ ${elements} != null && ${textures} != null && ${display} != null ]] || [[ ${parental} = null ]]
    do
       if [[ ${elements} = null ]]
       then
           elements="$(jq -rc '.elements' ${parental})"
           element_parent=${parental}
       fi
       if [[ ${textures} = null ]]
       then
           textures="$(jq -rc '.textures' ${parental})"
       fi
       if [[ ${display} = null ]]
       then
           display="$(jq -rc '.display' ${parental})"
       fi
       parental="$(jq -rc 'def namespace: if contains(":") then sub("\\:(.+)"; "") else "minecraft" end; ("./assets/" + (.parent | namespace) + "/models/" + ((.parent? // empty) | sub("(.*?)\\:"; "")) + ".json") // "null"' ${parental})"
    done

    if [[ ${elements} != null && ${textures} != null && ${display} != null ]]
    then
      matching_element="$(jq -rc --arg element_parent "${element_parent}" '(.[] | select(.element_parent == $element_parent or .path == $element_parent) | .geyserID) // "null"' config.json)"
      if [[ ${matching_element} = null ]]
      then
        jq -n -c --arg binding "c.item_slot == 'head' ? 'head' : q.item_slot_to_bone_name(c.item_slot)" --arg model_name "${gid}" --argjson jelements "${elements}" --argjson jtextures "${textures}" '{"textures": $jtextures, "elements": $jelements} |

        def element_array:
        (.textures | to_entries | sort_by(.key) | map({(.key): .value}) | add | keys_unsorted) as $texture_array
        | ($texture_array | length) as $frames
        | (($frames | sqrt) | ceil) as $sides
        | (.texture_size[1] // 16) as $t1
        | .elements | map({
          "origin": [(-.to[0] + 8), (.from[1]), (.from[2] - 8)],
          "size": [.to[0] - .from[0], .to[1] - .from[1], .to[2] - .from[2]],
          "rotation": (if (.rotation.axis) == "x" then [(.rotation.angle | tonumber * -1), 0, 0] elif (.rotation.axis) == "y" then [0, (.rotation.angle | tonumber * -1), 0] elif (.rotation.axis) == "z" then [0, 0, (.rotation.angle | tonumber)] else null end),
          "pivot": (if .rotation.origin then [(- .rotation.origin[0] + 8), .rotation.origin[1], (.rotation.origin[2] - 8)] else null end),
          "uv": (
            def uv_calc($input):
              (if (.faces | .[$input]) then
              (.faces | .[$input].texture[1:] as $input_n | $texture_array | (index($input_n) // index("particle"))) as $pos_n
              | ((.faces | .[$input].uv[0] / $sides) + ((fmod($pos_n; $sides)) * (16 / $sides))) as $fn0
              | ((.faces | .[$input].uv[1] / $sides) + ((($pos_n / $sides) | floor) * (16 / $sides))) as $fn1
              | ((.faces | .[$input].uv[2] / $sides) + ((fmod($pos_n; $sides)) * (16 / $sides))) as $fn2
              | ((.faces | .[$input].uv[3] / $sides) + ((($pos_n / $sides) | floor) * (16 / $sides))) as $fn3 |
              {
                "uv": [($fn0), ($fn1)],
                "uv_size": [($fn2 - $fn0), ($fn3 - $fn1)]
              } else null end);
            {
            "north": uv_calc("north"),
            "south": uv_calc("south"),
            "east": uv_calc("east"),
            "west": uv_calc("west"),
            "up": uv_calc("up"),
            "down": uv_calc("down")
            })
        }) | walk( if type == "object" then with_entries(select(.value != null)) else . end)
        ;

        def pivot_groups:
        (element_array) as $element_array |
        [[.elements[].rotation] | unique | .[] | select (.!=null)]
        | map((
        [(- .origin[0] + 8), .origin[1], (.origin[2] - 8)] as $i_piv |
        (if (.axis) == "x" then [(.angle | tonumber * -1), 0, 0] elif (.axis) == "y" then [0, (.angle | tonumber * -1), 0] else [0, 0, (.angle | tonumber)] end) as $i_rot |
        {
          "parent": "geysercmd_z",
          "pivot": ($i_piv),
          "rotation": ($i_rot),
          "mirror": true,
          "cubes": [($element_array | .[] | select(.rotation == $i_rot and .pivot == $i_piv))]
        }))
        ;

        {
          "format_version": "1.16.0",
          "minecraft:geometry": [{
            "description": {
              "identifier": ("geometry.geysercmd." + ($model_name)),
              "texture_width": 16,
              "texture_height": 16,
              "visible_bounds_width": 4,
              "visible_bounds_height": 4.5,
              "visible_bounds_offset": [0, 0.75, 0]
            },
            "bones": ([{
              "name": "geysercmd",
              "binding": $binding,
              "pivot": [0, 8, 0]
            }, {
              "name": "geysercmd_x",
              "parent": "geysercmd",
              "pivot": [0, 8, 0]
            }, {
              "name": "geysercmd_y",
              "parent": "geysercmd_x",
              "pivot": [0, 8, 0]
            }, {
              "name": "geysercmd_z",
              "parent": "geysercmd_y",
              "pivot": [0, 8, 0],
              "cubes": [(element_array | .[] | select(.rotation == null))]
            }] + (pivot_groups | map(del(.cubes[].rotation)) | to_entries | map( (.value.name = "rot_\(1+.key)" ) | .value)))
          }]
        }

        ' | sponge ./target/rp/models/blocks/geysercmd/${gid}.json

        jq --arg gid "${gid}" --arg element_parent "${element_parent}" '.[$gid] += {"element_parent": $element_parent}' ./config.json | sponge config.json
        geometryID=${gid}
        printf "\e[33m[•]\e[m \e[37mGenerated new geometry for ${geometryID}\e[m\n"
      else
        geometryID=${matching_element}
        printf "\e[33m[•]\e[m \e[37mUsing existing geometry from ${geometryID} for ${gid}\e[m\n"
        jq --arg gid "${gid}" --arg geometryID "${geometryID}" '.[$gid].geometry = ("geometry.geysercmd." + $geometryID)' config.json | sponge config.json
      fi
      # get texture array
      texture_array=($(jq -nr --argjson textures "${textures}" 'def namespace: if contains(":") then sub("\\:(.+)"; "") else "minecraft" end; {"textures": $textures} | .textures | to_entries | sort_by(.key) | map({(.key): .value}) | add | map("./assets/" + (. | namespace) + "/textures/" + (. | sub("(.*?)\\:"; "")) + ".png") | .[]'))
      # crop any mcmeta associated files on the first frame and make sure we have a texture
      tex_array_counter=0
      for i in "${texture_array[@]}"
      do
        if test -f "${i}.mcmeta"; then
            magick ${i} -background none -gravity North -extent "%[fx:h<w?h:w]x%[fx:h<w?h:w]" ${i}
            printf "\e[31m[!]\e[m \e[37mCropped .textures[${tex_array_counter}] for use in ${gid} becuase mcmeta was detected\e[m\n"
            rm "${i}.mcmeta"
        fi
        if ! test -f "${i}"; then
          texture_array[${tex_array_counter}]="./assets/minecraft/textures/fallbacktexture.png"
          printf "\e[31m[!]\e[m \e[37mFallback texture was used for ${gid}.textures[${tex_array_counter}]\e[m\n"
        fi
        tex_array_counter=$((tex_array_counter+1))
      done
      tile_dim=($(jq -nr --argjson textures "${textures}" '$textures | length | sqrt | ceil'))
      # find widest image & cap texture_width @ 128
      texture_width=($(identify -format "%w\n" ${texture_array[@]} | sort -n -r -k 1 | head -n 1))
      texture_width=$(($texture_width <= ${maximum_width_dimension} ? $texture_width : ${maximum_width_dimension}))
      # generate stitched texture with imagemagick (cap size @512) (perhaps we can filter to used textures only)
      montage ${texture_array[@]} -background transparent -geometry x${texture_width}+0+0 -tile ${tile_dim}x${tile_dim} -interpolate Integer -filter point ./target/rp/textures/blocks/geysercmd/${gid}.png
      # ensure we do not already have this texture, and reuse in the case that we do
      texture_id=${gid}
      texture_hash=($(md5 -q "./target/rp/textures/blocks/geysercmd/${gid}.png" 2>/dev/null))
      match_count=($(md5 -r ./target/rp/textures/blocks/geysercmd/*.png 2>/dev/null | grep "${texture_hash}" | wc -l))
      if [ ${match_count} -gt 1 ]
      then
          # remove our newly generated duplicate texture and change our texture_id to the match
          rm "./target/rp/textures/blocks/geysercmd/${gid}.png"
          texture_id=($(md5 -r ./target/rp/textures/blocks/geysercmd/*.png | grep "${texture_hash}" | cut -d "." -f 2 | cut -c38-))
          printf "\e[33m[•]\e[m \e[37mReusing texture from ${texture_id} for ${gid}\e[m\n"
      fi

      # append our texture information to rp terrain_texture.json
      jq -c --arg texture_id "${texture_id}" --arg geyser_id "${gid}" '
      .texture_data += {
        ($geyser_id): {
          "textures": ("textures/blocks/geysercmd/" + $texture_id)
        }
      }
      ' ./target/rp/textures/terrain_texture.json | sponge ./target/rp/textures/terrain_texture.json

      # now gen animation, block, and attachable, taking care to use our newly defined geometryID

      # generate our rp animations via display settings
      jq -c -n --argjson display "${display}" --arg model_name "${gid}" '{"display": $display} |
      {
        "format_version": "1.8.0",
        "animations": {
          ("animation.geysercmd." + ($model_name) + ".thirdperson_main_hand"): {
            "loop": true,
            "bones": {
              "geysercmd_x": (if .display.thirdperson_righthand then {
                "rotation": (if .display.thirdperson_righthand.rotation then [(- .display.thirdperson_righthand.rotation[0]), 0, 0] else null end),
                "position": (if .display.thirdperson_righthand.translation then [(- .display.thirdperson_righthand.translation[0]), (.display.thirdperson_righthand.translation[1]), (.display.thirdperson_righthand.translation[2])] else null end),
                "scale": (if .display.thirdperson_righthand.scale then [(.display.thirdperson_righthand.scale[0]), (.display.thirdperson_righthand.scale[1]), (.display.thirdperson_righthand.scale[2])] else null end)
              } else null end),
              "geysercmd_y": (if .display.thirdperson_righthand.rotation then {
                 "rotation": (if .display.thirdperson_righthand.rotation then [0, (- .display.thirdperson_righthand.rotation[1]), 0] else null end)
              } else null end),
              "geysercmd_z": (if .display.thirdperson_righthand.rotation then {
                 "rotation": [0, 0, (.display.thirdperson_righthand.rotation[2])]
              } else null end),
              "geysercmd": {
                 "rotation": [90, 0, 0],
                 "position": [0, 13, -3]
              }
            }
          },
          ("animation.geysercmd." + ($model_name) + ".thirdperson_off_hand"): {
            "loop": true,
            "bones": {
              "geysercmd_x": (if .display.thirdperson_lefthand then {
                "rotation": (if .display.thirdperson_lefthand.rotation then [(- .display.thirdperson_lefthand.rotation[0]), 0, 0] else null end),
                "position": (if .display.thirdperson_lefthand.translation then [(- .display.thirdperson_lefthand.translation[0]), (.display.thirdperson_lefthand.translation[1]), (.display.thirdperson_lefthand.translation[2])] else null end),
                "scale": (if .display.thirdperson_lefthand.scale then [(.display.thirdperson_lefthand.scale[0]), (.display.thirdperson_lefthand.scale[1]), (.display.thirdperson_lefthand.scale[2])] else null end)
              } else null end),
              "geysercmd_y": (if .display.thirdperson_lefthand.rotation then {
                 "rotation": (if .display.thirdperson_lefthand.rotation then [0, (- .display.thirdperson_lefthand.rotation[1]), 0] else null end)
              } else null end),
              "geysercmd_z": (if .display.thirdperson_lefthand.rotation then {
                 "rotation": [0, 0, (.display.thirdperson_lefthand.rotation[2])]
              } else null end),
              "geysercmd": {
                 "rotation": [90, 0, 0],
                 "position": [0, 13, -3]
              }
            }
          },
          ("animation.geysercmd." + ($model_name) + ".head"): {
            "loop": true,
            "bones": {
              "geysercmd_x": {
                "rotation": (if .display.head.rotation then [(- .display.head.rotation[0]), 0, 0] else null end),
                "position": (if .display.head.translation then [(- .display.head.translation[0] * 0.625), (.display.head.translation[1] * 0.625), (.display.head.translation[2] * 0.625)] else null end),
                "scale": (if .display.head.scale then (.display.head.scale | map(. * 0.625)) else 0.625 end)
              },
              "geysercmd_y": (if .display.head.rotation then {
                "rotation": [0, (- .display.head.rotation[1]), 0]
              } else null end),
              "geysercmd_z": (if .display.head.rotation then {
                "rotation": [0, 0, (.display.head.rotation[2])]
              } else null end),
              "geysercmd": {
                 "position": [0, 19.5, 0]
              }
            }
          },
          ("animation.geysercmd." + ($model_name) + ".firstperson_main_hand"): {
            "loop": true,
            "bones": {
              "geysercmd": {
                "rotation": [90, 60, -40],
                "position": [4, 10, 4],
                "scale": 1.5
              },
              "geysercmd_x": {
                "position": (if .display.firstperson_righthand.translation then [(- .display.firstperson_righthand.translation[0]), (.display.firstperson_righthand.translation[1]), (- .display.firstperson_righthand.translation[2])] else null end),
                "rotation": (if .display.firstperson_righthand.rotation then [(- .display.firstperson_righthand.rotation[0]), 0, 0] else [0.1, 0.1, 0.1] end),
                "scale": (if .display.firstperson_righthand.scale then (.display.firstperson_righthand.scale) else null end)
              },
              "geysercmd_y": (if .display.firstperson_righthand.rotation then {
                "rotation": [0, (- .display.firstperson_righthand.rotation[1]), 0]
              } else null end),
              "geysercmd_z": (if .display.firstperson_righthand.rotation then {
                "rotation": [0, 0, (.display.firstperson_righthand.rotation[2])]
              } else null end)
            }
          },
          ("animation.geysercmd." + ($model_name) + ".firstperson_off_hand"): {
            "loop": true,
            "bones": {
              "geysercmd": {
                "rotation": [90, 60, -40],
                "position": [4, 10, 4],
                "scale": 1.5
              },
              "geysercmd_x": {
                "position": (if .display.firstperson_lefthand.translation then [(.display.firstperson_lefthand.translation[0]), (.display.firstperson_lefthand.translation[1]), (- .display.firstperson_lefthand.translation[2])] else null end),
                "rotation": (if .display.firstperson_lefthand.rotation then [(- .display.firstperson_lefthand.rotation[0]), 0, 0] else [0.1, 0.1, 0.1] end),
                "scale": (if .display.firstperson_lefthand.scale then (.display.firstperson_lefthand.scale) else null end)
              },
              "geysercmd_y": (if .display.firstperson_lefthand.rotation then {
                "rotation": [0, (- .display.firstperson_lefthand.rotation[1]), 0]
              } else null end),
              "geysercmd_z": (if .display.firstperson_lefthand.rotation then {
                "rotation": [0, 0, (.display.firstperson_lefthand.rotation[2])]
              } else null end)
            }
          }
        }
      } | walk( if type == "object" then with_entries(select(.value != null)) else . end)

      ' | sponge ./target/rp/animations/geysercmd/animation.${gid}.json

      # generate our rp attachable definition
      jq -c -n --arg attachable_material "${attachable_material}" --arg v_main "v.main_hand = c.item_slot == 'main_hand';" --arg v_off "v.off_hand = c.item_slot == 'off_hand';" --arg v_head "v.head = c.item_slot == 'head';" --arg geometryID "${geometryID}" --arg model_name "${gid}" --arg texture_name "${texture_id}" '

      {
        "format_version": "1.10.0",
        "minecraft:attachable": {
          "description": {
            "identifier": ("geysercmd:" + $model_name),
            "materials": {
              "default": $attachable_material,
              "enchanted": $attachable_material
            },
            "textures": {
              "default": ("textures/blocks/geysercmd/" + $texture_name),
              "enchanted": "textures/misc/enchanted_item_glint"
            },
            "geometry": {
              "default": ("geometry.geysercmd." + $geometryID)
            },
            "scripts": {
              "pre_animation": [$v_main, $v_off, $v_head],
              "animate": [
                {"thirdperson_main_hand": "v.main_hand && !c.is_first_person"},
                {"thirdperson_off_hand": "v.off_hand && !c.is_first_person"},
                {"thirdperson_head": "v.head && !c.is_first_person"},
                {"firstperson_main_hand": "v.main_hand && c.is_first_person"},
                {"firstperson_off_hand": "v.off_hand && c.is_first_person"},
                {"firstperson_head": "c.is_first_person && v.head"}
              ]
            },
            "animations": {
              "thirdperson_main_hand": ("animation.geysercmd." + $model_name + ".thirdperson_main_hand"),
              "thirdperson_off_hand": ("animation.geysercmd." + $model_name + ".thirdperson_off_hand"),
              "thirdperson_head": ("animation.geysercmd." + $model_name + ".head"),
              "firstperson_main_hand": ("animation.geysercmd." + $model_name + ".firstperson_main_hand"),
              "firstperson_off_hand": ("animation.geysercmd." + $model_name + ".firstperson_off_hand"),
              "firstperson_head": "animation.geysercmd.disable"
            },
            "render_controllers": [ "controller.render.item_default" ]
          }
        }
      }

      ' | sponge ./target/rp/attachables/geysercmd/${gid}.attachable.json

      # generate our bp block definition
      jq -c -n --arg block_material "${block_material}" --arg geometryID "${geometryID}" --arg geyser_id "${gid}" '

      {
          "format_version": "1.16.200",
          "minecraft:block": {
              "description": {
                  "identifier": ("geysercmd:" + $geyser_id)
              },
              "components": {
                  "minecraft:material_instances": {
                      "*": {
                          "texture": $geyser_id,
                          "render_method": $block_material,
                          "face_dimming": false,
                          "ambient_occlusion": false
                      }
                  },
                  "tag:geysercmd:example_block": {},
                  "minecraft:geometry": ("geometry.geysercmd." + $geometryID),
                  "minecraft:placement_filter": {
                    "conditions": [
                        {
                            "allowed_faces": [
                            ],
                            "block_filter": [
                            ]
                        }
                    ]
                  }
              }
          }
      }

      ' | sponge ./target/bp/blocks/geysercmd/${gid}.json
      printf "\e[32m[+]\e[m \e[37mChild ${gid} converted\e[m\n"
      ProgressBar ${cur_pos} ${_end}
      echo
    else
      printf "\e[37mSuitable parent information was not availbile for ${gid}...\e[m\n"
      printf "\e[31m[X]\e[m \e[37mDeleting ${gid} from config\e[m\n"
      ProgressBar ${cur_pos} ${_end}
      echo
      jq --arg gid "${gid}" 'del(.[$gid])' config.json | sponge config.json
    fi
done < pa.csv

printf "\e[32m[+]\e[m \e[37mFinished conversion of child models\e[m\n"
echo

# write lang file US
printf "\e[33m[•]\e[m \e[37mWriting en_US and en_GB lang files\e[m\n"
mkdir ./target/rp/texts
jq -r '

def format: (.[0:1] | ascii_upcase ) + (.[1:] | gsub( "_(?<a>[a-z])"; (" " + .a) | ascii_upcase));
to_entries[]|"\("tile.geysercmd:" + .key + ".name")=\(.value.item | format)"

' config.json | sponge ./target/rp/texts/en_US.lang

# copy US lang to GB
cp ./target/rp/texts/en_US.lang ./target/rp/texts/en_GB.lang

# write supported languages file
jq -n '["en_US","en_GB"]' | sponge ./target/rp/texts/languages.json
printf "\e[32m[+]\e[m \e[37men_US and en_GB lang files written\e[m\n"
echo

# apply image compression if we can
if command -v convert >/dev/null 2>&1 ; then
    printf "\e[32m[+]\e[m \e[37mOptional dependency pngquant detected\e[m\n"
    printf "\e[33m[•]\e[m \e[37mAttempting image compression\e[m\n"
    pngquant -f --skip-if-larger --ext .png --strip ./target/rp/textures/blocks/geysercmd/*.png
    printf "\e[32m[+]\e[m \e[37mImage compression complete\e[m\n"
    echo
fi

# attempt to merge with existing pack if input was provided
if test -f ${merge_input}; then
  mkdir inputbedrockpack
  printf "\e[33m[•]\e[m \e[37mDecompressing input bedrock pack\e[m\n"
  unzip -q ${merge_input} -d ./inputbedrockpack
  printf "\e[33m[•]\e[m \e[37mMerging input bedrock pack with generated bedrock assets\e[m\n"
  cp -n -r "./inputbedrockpack"/* './target/rp/'
  if test -f ./inputbedrockpack/textures/terrain_texture.json; then
    printf "\e[33m[•]\e[m \e[37mMerging terrain texture files\e[m\n"
    jq -s '
    {"resource_pack_name": "vanilla",
    "texture_name": "atlas.terrain",
    "padding": 8, "num_mip_levels": 4,
    "texture_data": (.[1].texture_data + .[0].texture_data)}
    ' ./target/rp/textures/terrain_texture.json ./inputbedrockpack/textures/terrain_texture.json | sponge ./target/rp/textures/terrain_texture.json
  fi
  if test -f ./inputbedrockpack/texts/languages.json; then
    printf "\e[33m[•]\e[m \e[37mMerging languages file\e[m\n"
    jq -s '.[0] + .[1] | unique' | sponge ./target/rp/texts/languages.json
  fi
  if test -f ./inputbedrockpack/texts/en_US.lang; then
    printf "\e[33m[•]\e[m \e[37mMerging en_US lang file\e[m\n"
    cat ./inputbedrockpack/texts/en_US.lang >> ./target/rp/texts/en_US.lang
  fi
  if test -f ./inputbedrockpack/texts/en_GB.lang; then
    printf "\e[33m[•]\e[m \e[37mMerging en_GB lang file\e[m\n"
    cat ./inputbedrockpack/texts/en_GB.lang >> ./target/rp/texts/en_GB.lang
  fi
  printf "\e[31m[X]\e[m \e[37mDeleting input bedrock pack scratch direcotry\e[m\n"
  rm -rf inputbedrockpack
  printf "\e[32m[+]\e[m \e[37mInput bedrock pack merged with generated assets\e[m\n"
  echo
fi

# cleanup
printf "\e[31m[X]\e[m \e[37mDeleting scratch files\e[m\n"
rm -rf assets && rm -f pack.mcmeta && rm -f pack.png && rm -f parents.json && rm -f np.csv && rm -f pa.csv && rm -f README.md && rm -f README.txt

printf "\e[31m[X]\e[m \e[37mDeleting unused entries from config\e[m\n"
jq 'map_values(del(.path, .element_parent, .parent, .geyserID))' config.json | sponge config.json
printf "\e[33m[•]\e[m \e[37mMoving config to target directory\e[m\n"
mv config.json ./target/config.json
echo

printf "\e[33m[•]\e[m \e[37mCompressing output packs\e[m\n"
mkdir ./target/packaged
cd ./target/rp > /dev/null && zip -rq8 geyser_rp.mcpack . -x "*/.*" && cd - > /dev/null && mv ./target/rp/geyser_rp.mcpack ./target/packaged/geyser_rp.mcpack
cd ./target/bp > /dev/null && zip -rq8 geyser_bp.mcpack . -x "*/.*" && cd - > /dev/null && mv ./target/bp/geyser_bp.mcpack ./target/packaged/geyser_bp.mcpack
cd ./target/packaged > /dev/null && zip -rq8 geyser.mcaddon . -i "*.mcpack" && cd - > /dev/null
mkdir ./target/unpackaged
mv ./target/rp ./target/unpackaged/rp && mv ./target/bp ./target/unpackaged/bp

echo
printf "\e[32m[+]\e[m \e[1m\e[37mConversion Process Complete\e[m\n"
echo
printf "\e[37mExiting...\e[m\n"
echo
