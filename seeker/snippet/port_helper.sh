#date: 2023-06-30T16:57:24Z
#url: https://api.github.com/gists/52edf14618f128a5340c275eba316808
#owner: https://api.github.com/users/i30817

#!/usr/bin/bash

echo "Script to help port Shadowrun Dragonfall UGCs to the Shadowrun Hong Kong engine, use it on the source directory of a dragonfall UGC. NOT a Hong Kong or SRR UGC and only the source not the compiled version"

[ ! -f "./project.cpack.txt" ] && { echo "Not called on a UGC source directory"; exit 0; }

grep -l 'project_name: "CFiC"' ./project.cpack.txt && {

echo "\

CalFree in chains has the idea of replacing the ORIGINAL music by their own versions of the soundtrack with a utility and using those original names. I on the other hand have the idea of adding the dragonfall soundtrack with the shadowed utility ( go install github.com/betrok/shadowed@latest ) so all the ported mods music works seamlessly. This doesn't actually affect callfree in chains because the utility they recommend backups the old music and can restore it, but i still find it distasteful to have to use other utilities, so my mod music includes the callfree in chains sound tracks with other names, so i can replace those references in a custom version of the UGC. This won't affect others.
"
echo -e $'Press any key to replace the calfree in chains music names or Ctrl+C to exit...\n' 
read -rs -n1


declare -a CALFREMUSIC=(Sewer loudmusic Combat-Boss Hub-Exterior Hub-TeaHouse Legwork-Erhu Legwork-News Hub-SafeHouse TitleTheme-UI Combat-Matrix2 Legwork-Gobbet Legwork-Is0bel Legwork-Museum Legwork-Generic Legwork-Grendel Legwork-Hacking Legwork-Kowloon Stealth-Matrix1 Combat-Gobbet-Int1 Combat-Gobbet-Int2 Combat-Is0bel-Int1 Combat-Is0bel-Int2 Combat-stinger-end Legwork-SLinterior Combat-Generic-Int1 Combat-Generic-Int2 Combat-Grendel-Int1 Combat-Grendel-Int2 Combat-Kowloon-Int1 Combat-Kowloon-Int2 Hub-Club88-InStreet Hub-Club88-MainRoom KnightKingsElevator Combat-Gobbet-WrapUp Combat-Is0bel-WrapUp Combat-stinger-start Combat-Generic-WrapUp Combat-Grendel-WrapUp Combat-Kowloon-WrapUp Legwork-ExitStageLeft Legwork-Whistleblower Legwork-VictoriaHarbor Hub-Club88-ThroughWalls Combat-VictoriaHarbor-Int1 Combat-VictoriaHarbor-Int2 Combat-VictoriaHarbor-WrapUp)

grep -rl "\"TESTSTINGER\"" . | tr '\n' '\0' | xargs -r --null -n1 sed -i 's/"TESTSTINGER"/"CFIC-TESTSTINGER"/g'
for i in "${CALFREMUSIC[@]}"
do
   grep -rl "\"HongKong-$i\"" . | tr '\n' '\0' | xargs -r --null -n1 sed -i "s/\"HongKong-$i\"/\"CFIC-$i\"/g"
done

exit 0
}


echo ""
echo "Before running this you should first add the compatibility UGC Dragonfall2HongKong as the last of the Content Pack dependencies on the editor, and assuming it worked, try to extract the source. Assuming it worked again (sometimes data files aren't extracted from the zip but can be extracted manually for some reason i don't understand), then edit the project.cpack.txt file to set read_only to false and you can run this."

echo -e $'Press any key to continue or Ctrl+C to exit...\n' 
read -rs -n1


echo "\
First incompatibility is that the datajack slot name changed from 'cyberware_jack' to 'cyberware_head'. That can be automatically corrected, although if the editor considers those old datajacks 'obsolete' they won't appear on the editor GUI slot (but they do work in game).

"

files=$(grep -rl 'cyberware_jack:' . | cut -f 1 -d ':' | sort --unique)
echo -n "$files" | tr '\n' '\0' | xargs -r --null -n1 sed -i 's/cyberware_jack:/cyberware_head:/g'


echo "\
Second incompatibility is 2 triggers were deprecated, and can even crash the editor if you open a map with those triggers, so you need to manually correct them after extracting the source.

'Enable/Disable Manual Turn Mode' and 'Evaluate Turn Mode' need to turn into versions with ' in Dimension' at the end of 'functionName' and another final argument whose name might vary, but is almost always:
      args {
        call_value {
          functionName: \"Get Map Item (SceneDimension)\"
          args {
            string_value: \"Default\"
          }
        }
      }
      
This will open the default text GUI editor with them so you can search for them and replace after checking the surrounding code to see if it's the right dimension (it always is afaict). Note that these triggers are used to force turn based mode on missions, sometimes when interacting with items, but sometimes at the start of missions. I don't actually like forcing TB mode and have a workaround for it in my executable hack (ctrl asks any deckers to hack the nearest matrix junction), so sometimes i just remove all the surrounding triggers on my ports.

"

problems=$(find .  -not -path '*/.*'  -type f \( -wholename '*/scenes/*.srt.txt' -o -wholename '*/convos/*.convo.txt' \) -exec grep -lZE 'functionName: "(Enable/Disable Manual|Evaluate) Turn Mode"' {} +)

echo -n "$problems" | sort -z --unique | xargs --null -r -n1 xdg-open

if [ ! -z "$problems" ]; then
  echo "If you think you're done when you finish that, think again, because I will open more files only once those problems are resolved"
  exit 0
fi

echo "\
Third incompatibility is that matrix cameras now need that fact on a property. This will indicate which files have that problem by searching for 'scene_dimensions' and counting them, then subtracting the number of 'is_matrix: true'. If the subtraction is not equal 1, there are problems because either there is one normal dimension and no matrix dimension, or 1 normal dimension and N matrix dimensions - there is no other reason to have dimensions. 

Note that areas without a dedicated camera are problematic because it can lead to infinite combat turns from enemies - that mostly appears to happen if you try to convert a DMS mod though, since Dragonfall already required cameras.

"

problems1=$(find .  -not -path '*/.*' -type f -wholename '*/scenes/*.srt.txt')
problems2=$(find .  -not -path '*/.*' -type f -wholename '*/scenes/*.srt.txt' -exec grep -hc 'scene_dimensions {' {} +)
problems3=$(find .  -not -path '*/.*' -type f -wholename '*/scenes/*.srt.txt' -exec grep -hc 'is_matrix: true' {} +)

problems=$(paste <(echo -n "$problems2") <(echo -n "$problems3") <(echo -n "$problems1"))

problems=$(awk '$1 - $2 != 1' <(echo -n "$problems"))
echo "suspect files:"
echo -e "nºdim\tnºmtxdim\tfile"
echo "$problems"
cut -f3 <(echo -n "$problems") | tr '\n' '\0' | xargs -r --null -n1 -t xdg-open

if [ ! -z "$problems" ]; then
  echo "If you think you're done when you finish that, think again, because I will open more files only once those problems are resolved"
  exit 0
fi

echo "\
Fourth incompatibility is the broken orcs, trolls new 'core' meshes incompatibility with deprecated outfits - which unfortunately includes some disguises. Core meshes are like the player meshes in that they can adapt to several outfits instead of being fixed, and both of these are categories in the inventory choice widget. Unfortunately, there are a bunch of older outfits that were not completed changed for the new ork core meshes to work on them, and the game categorically refuses to load the older meshes.

The possible solutions are some madlad going through all of the broken outfits and changing the meshes to adapt to the new core meshes then reinserting them into the assets (like shadowed can do for the music), or change the outfits by hand for the trolls and orks into 'the closest viable one'. Note that not all of meshes appear broken in all outfits. For instance, troll female security seems ok, but male is broken.
"

readarray -d $'\n' -t maybe_need_replacement < <(grep -rn -E "prefab_name: \"(Seattle:)?Core/(Troll|Ork)(Fe)?(m|M)ale\"" . | cut -f1,2 -d ":" )

the hardway
declare -A OUTFITS=( 
["\"AdeptCombatvest\""]="NPC_HK_Adept_StreetBrawler"
["\"AdeptKunai\""]="NPC_HK_Adept_KunaiSuit"
["\"AdeptNinja\""]="Player_AdeptNinja"
["\"AdeptStealth\""]="Player_AdeptStealth"
["\"AdeptStreetMonk\""]="NPC_HK_Adept_StreetMonk"
["\"AdversaryCultist\""]="NONE"
["\"Aztechnology\""]="NONE SUGGEST Model Berlin:Aztechnology/*"
["\"Aztechnology Disguise\""]="NONE SUGGEST Model Berlin:Aztechnology/*"
["\"Knight Errant Soldier Disguise\""]="NONE SUGGEST MODEL Berlin:KnightErrants/*"
["\"DeckerJacket\""]="Player_DeckerJacket"
["\"DeckerStreet\""]="Player_DeckerStreet"
["\"Knight Errant Soldier\""]="Knight Errant Soldier Disguise"
["\"LatticeJacket\""]="Player_Outfit_Tech1"
["\"MageCasual\""]="Player_MageCasual"
["\"MageDark\""]="Player_MageDark"
["\"MageRedRidingHood\""]="Player_MageRedRidingHood"
["\"MageSlick\""]="Player_MageSlick"
["\"MageTraditional\""]="Player_MageTraditional"
["\"Outfit_Corporate Salaryman\""]="Player_HK_Suit"
["\"Outfit_Cult Follower\""]="NONE"
["\"Outfit_Maintenance\""]="Tsang Disguise"
["\"Outfit_Scientist\""]="NONE SUGGEST Ork Male Scientist Mask MODEL,OrkMaleTsangScientist MODEL"
["\"Outfit_Security\""]="TrollFemale OK SUGGEST OrkMaleSecurityGuard MODEL,TrollMaleSecurityGuard MODEL"
["\"Outfit_SINless\""]="NONE"
["\"RiggerFlightsuit\""]="Player_RiggerFlightsuit"
["\"RiggerGolden\""]="NPC_HK_Tech_RiggerGolden"
["\"RiggerHawaiienShirt\""]="Player_RiggerHawaiianShirt"
["\"RiggerToolbelt\""]="Player_Outfit_Tech2Drone"
["\"RiggerTrenchcoat\""]="NPC_HK_Tech_RiggerTrench"
["\"SamuraiBunny\""]="Player_SamuraiBunny"
["\"SamuraiBunnyMaskless\""]="Player_Outfit_Combat25Melee"
["\"SamuraiMilitary\""]="Player_Outfit_Combat3DPS"
["\"SamuraiPunk\""]="Player_SamuraiPunk"
["\"SamuraiTrenchcoat\""]="NPC_HK_Combat_SecureTrench"
["\"ShamanPendant\""]="Player_ShamanPendant"
["\"ShamanSkirted\""]="Player_ShamanSkirted"
["\"ShamanTotemCoat\""]="Player_ShamanTotemCoat"
["\"ShamanUrban\""]="Player_ShamanUrban"
)

deprecated=""
for key in "${!OUTFITS[@]}"
do
  deprecated+="$key|"
done
deprecated="(${deprecated%\|})"

readarray -d $'\n' -t maybe_need_replacement < <(grep -rn -E "prefab_name: \"(Seattle:)?Core/(Troll|Ork)(Fe)?(m|M)ale\"" . | cut -f1,2 -d ":" )

for t in "${maybe_need_replacement[@]}"
do
  filepart=$(echo -n "$t" | cut -f1 -d ":")
  linepart=$(echo -n "$t" | cut -f2 -d ":")
  offset=$(tail -n +"$linepart" "$filepart" | grep -m1 -n -E "^}" | cut -f1 -d ":")
  if match=$(tail -n +"$linepart" "$filepart" | head -n "$offset" | grep -m1 -o -E "$deprecated"); then
    echo "$match suggested replacement \"${OUTFITS[$match]}\" \"$filepart\""
    gedit "$filepart" "+$linepart"
  fi
done