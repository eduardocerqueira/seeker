#date: 2024-04-30T16:46:09Z
#url: https://api.github.com/gists/4d6093ee7179d7b3d9ce37bb71254e77
#owner: https://api.github.com/users/web3recon

#! /bin/bash
#
# Diffusion youtube avec ffmpeg

# Configurer youtube avec une résolution 720p. La vidéo n'est pas scalée.

VBR="2500k"                                    # Bitrate de la vidéo en sortie
FPS="30"                                       # FPS de la vidéo en sortie
QUAL="medium"                                  # Preset de qualité FFMPEG
YOUTUBE_URL="rtmp://a.rtmp.youtube.com/live2"  # URL de base RTMP youtube

FOLDER="videos"                                    # Source UDP (voir les annonces SAP)
KEY="...."                                     # Clé à récupérer sur l'event youtube

SOURCE=""
n=0
filter=""

for f in $FOLDER/*.mp4
do
  SOURCE="$SOURCE -i $f"
  filter="$filter [$n:v:0] [$n:a:0]"
  ((n++))
done

filter="$filter concat=n=$n:v=1:a=1 [v] [a]"

echo "ffmpeg $SOURCE -filter_complex '$filter'"

ffmpeg \
    $SOURCE -filter_complex "$filter" \
-map "[v]" -map "[a]" -deinterlace \
    -vcodec libx264 -pix_fmt yuv420p -preset $QUAL -r $FPS -g $(($FPS * 2)) -b:v $VBR \
    -acodec libmp3lame -ar 44100 -threads 6 -qscale 3 -b:a 712000 -bufsize 512k \
    -f flv "$YOUTUBE_URL/$KEY"
