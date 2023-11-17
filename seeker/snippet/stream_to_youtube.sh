#date: 2023-11-17T16:35:19Z
#url: https://api.github.com/gists/77db1b230f488fa8b36f09e9ab32c0af
#owner: https://api.github.com/users/softwaretro

#! /bin/bash
#
# Diffusion youtube avec ffmpeg

# Configurer youtube avec une résolution 720p. La vidéo n'est pas scalée.

VBR="2500k"                                    # Bitrate de la vidéo en sortie
FPS="30"                                       # FPS de la vidéo en sortie
QUAL="medium"                                  # Preset de qualité FFMPEG
YOUTUBE_URL="rtmp://a.rtmp.youtube.com/live2"  # URL de base RTMP youtube

SOURCE="<File_Location.ext>"              # Source UDP (voir les annonces SAP)
KEY="<Your_Stream_Key>"                                     # Clé à récupérer sur l'event youtube

ffmpeg \
    -stream_loop -1 -i "$SOURCE" -deinterlace \
    -vcodec libx264 -pix_fmt yuv420p -preset $QUAL -r $FPS -g $(($FPS * 2)) -b:v $VBR \
    -acodec libmp3lame -ar 44100 -threads 6 -qscale 3 -b:a 712000 -bufsize 512k \
    -f flv "$YOUTUBE_URL/$KEY"
