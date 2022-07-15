#date: 2022-07-15T17:22:44Z
#url: https://api.github.com/gists/b060edb3e7e9ac242ee1e45ea0bb8b12
#owner: https://api.github.com/users/dvessel

#!/bin/zsh

if [[ $argv[(I)(--help|-h)] -gt 0 ]]; then
  echo "
  --dtm|-p replay.dtm
    Path to dtm replay file. This is required.
  --game|-g game.gcz
    Path to rom file. Path will be detected automatically for some file types
    (rvz|iso|wia). For others, the path must be set manually.
  --dolphin|-o /Application/Dolphin.app
    Path to Dolphin.app. Defaults to Application folder.
  --repeat|-r 3
    How many run for each graphics backend. Defaults to 3.
  --user=~/Library/Application\ Support/Dolphin|-u ~path
    Path to support folder. Defaults to user library folder.
  "
  exit
fi

dtmpath=${argv[( $argv[(i)--dtm|-p] + 1 )]##-*}
if [[ ! -f $dtmpath ]]; then
  echo "[--dtm file.dtm] not set or the file doesn't exist." >&2
  exit 1
fi

gameid6=`xxd -seek 4 -len 6 $dtmpath | xxd -rp -seek -4`
gamepath=${argv[( $argv[(i)--game|-g] + 1 )]##-*}
reploop=${${argv[( $argv[(i)--repeat|-r] + 1 )]##-*}:-3}

dolphin=${${argv[( $argv[(i)--dolphin|-o] + 1 )]##-*}:-/Applications/Dolphin.app}
if [[ ! -d $dolphin ]]; then
  echo "$dolphin not found. Set --dolphin with a path to the application." >&2
  exit 1
fi

# Follow --user= path.
usr_default=~/Library/Application\ Support/Dolphin
usr_dir=${${argv[$argv[(i)--user=*]]##--user=}:-${argv[( $argv[(i)-u] + 1 )]##-*}}
usr_dir=${usr_dir:-$usr_default}
if [[ ! -d $usr_dir ]]; then
  echo "$usr_dir not found. Set --user with a path to the application." >&2
  exit 1
fi

if [[ -z $gamepath ]]; then
  while read -r p; do
    if [[ $p =~ (/.*) && -d $match[1] ]]; then
      isopath=$match[1]
      while read -r g; do
        if [[ `file -b $g` =~ $gameid6,.*\)$ ]]; then
          gamepath=$g
          break 2
        fi
      done < <( find -E $isopath -regex ".+\.(rvz|iso|wia)" )
    fi
  done < <( grep "ISOPath[0-9]" $usr_dir/Config/Dolphin.ini )
  if [[ -z $gamepath ]]; then
    echo "Game not found for $gameid6. Set it with --game pathto/game.xyz" >&2; exit 1
  fi
else
  if [[ ! -f $gamepath ]]; then
    echo "File not found: --game $gamepath" >&2; exit 1
  fi
fi

# Patch the .dtm file since the backend is written to it.
# This will only be a problem if a game depends on EFB CPU access such
# as Metroid Prime 2 or Super Mario Galaxy. Does not affect most games.
typeset -A hexsplice=(
  metal  "00000051: 4d65 7461 6c00                           Metal."
  vulkan "00000051: 5675 6c6b 616e                           Vulkan"
  opengl "00000051: 4f47 4c00 0000                           OGL..."
)

render_time="$usr_dir/Logs/render_time.txt"
output_logs="$dtmpath:r `date "+%m.%d,%H.%M.%S"`"
mkdir -p $output_logs

# Clean-up after manual interrupt.
trap 'rm -f $dtmpath:r-$backend.dtm
      rm -f $dtmpath:r-$backend.dtm.sav
      exit' INT

for backend in metal vulkan opengl; do

  head -c 81 $dtmpath > $dtmpath:r-$backend.dtm
  printf $hexsplice[$backend] | xxd -rp >> $dtmpath:r-$backend.dtm
  tail -c +88 $dtmpath >> $dtmpath:r-$backend.dtm
  if [[ -f $dtmpath.sav ]]; then
    ln -sf $dtmpath:t.sav $dtmpath:r-$backend.dtm.sav
  fi

  if [[ $backend == vulkan ]]; then
    export MVK_CONFIG_LOG_LEVEL=2
  fi

  for i in {1..$reploop}; do
    rm -f $render_time

    $dolphin/Contents/MacOS/Dolphin \
      --batch \
      --exec=$gamepath \
      --movie=$dtmpath:r-$backend.dtm \
      --config=Graphics.Settings.LogRenderTimeToFile=True \
      --config=Graphics.Settings.ShowFPS=True \
      --config=Dolphin.Core.EmulationSpeed=0 \
      --config=Dolphin.DSP.Backend=No\ Audio\ Output \
      --config=Dolphin.General.ShowFrameCount=True \
      --config=Dolphin.Movie.PauseMovie=True \
      --config=Logger.Options.WriteToConsole=False \
    & while [[ ! -f $render_time ]] sleep 2

    if [[ -f $render_time ]]; then
      printf "////////// $backend run $i/$reploop"
      m=. l=,
      while [[ $m != $l ]]; do
        printf .; sleep 2
        l=$m; m=`date -r $render_time +%s`
      done

      result=$output_logs/$dtmpath:t:r-$backend-$i.txt
      tail -n +60 $render_time > $result
      echo "\n////////// Sampled `wc -l $result` frames."
    else
      echo "////////// $backend $i/$reploop did not run\!" >&2
    fi

    killall Dolphin; sleep 2
  done

  rm -f $dtmpath:r-$backend.dtm
  rm -f $dtmpath:r-$backend.dtm.sav
done

echo "Saved: $output_logs"
