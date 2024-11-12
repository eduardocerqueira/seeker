#date: 2024-11-12T16:57:27Z
#url: https://api.github.com/gists/383d8d2a0bb17ef4ed5735716b2169d1
#owner: https://api.github.com/users/birabittoh

#!/bin/bash

# Funzione per controllare se un comando è disponibile
check_dependency() {
    command -v "$1" >/dev/null 2>&1 || { echo "$1 non è installato. Per favore installalo."; exit 1; }
}

# Controllo delle dipendenze
check_dependency "ffmpeg"
check_dependency "bc"

# Impostazioni di default
audio_file="song.opus"
video_file="guess.mp4"
output_file="output.mp4"
skip=0       # Tempo di partenza della riproduzione audio
fade=1       # Durata di fade-in e fade-out
delay=3      # Ritardo prima dell'inizio dell'audio
duration=14  # Durata della riproduzione di song.opus

# Funzione di aiuto
usage() {
    echo "Uso: $0 [opzioni]"
    echo "Opzioni:"
    echo "  -a FILE   file audio di input (default: song.opus)"
    echo "  -v FILE   file video di input (default: guess.mp4)"
    echo "  -o FILE   file video di output (default: output.mp4)"
    echo "  -t FLOAT  tempo di partenza (s) della riproduzione audio (default: 0)"
    echo "  -f FLOAT  durata (s) di fade-in e fade-out (default: 1)"
    echo "  -d FLOAT  attesa (s) prima dell'inizio dell'audio (default: 3)"
    echo "  -s FLOAT  durata (s) della riproduzione di song.opus (default: 14)"
    echo "  -h        mostra questo testo di aiuto"
    exit 1
}

# Parsing degli argomenti
while getopts "a:v:o:t:f:d:s:h" opt; do
    case $opt in
        a) audio_file="$OPTARG" ;;
        v) video_file="$OPTARG" ;;
        o) output_file="$OPTARG" ;;
        t) skip="$OPTARG" ;;
        f) fade="$OPTARG" ;;
        d) delay="$OPTARG" ;;
        s) duration="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Convertiamo delay in millisecondi e lo arrotondiamo
delay_ms=$(printf "%.0f" $(echo "$delay * 1000" | bc))

# Calcola il punto di inizio del fade-out in funzione di S
fade_out_start=$(echo "$duration - $fade" | bc)

# Definisce il filtro audio con fade-out e, se necessario, aggiunge il fade-in
audio_filter="afade=t=out:st=${fade_out_start}:d=$fade"
if (( $(echo "$skip >= $fade" | bc -l) )); then
    audio_filter="afade=t=in:ss=0:d=$fade,$audio_filter"
fi

# Comando ffmpeg
ffmpeg -i "$video_file" -i "$audio_file" -filter_complex "
[1:a]atrim=start=$skip:end=$(echo "$skip + $duration" | bc),asetpts=PTS-STARTPTS,${audio_filter},adelay=${delay_ms}|${delay_ms}[song];
[0:a][song]amix=inputs=2:duration=first[audio_mix];
[0:v]copy[v]" -map "[v]" -map "[audio_mix]" -c:v libx264 -c:a aac -shortest "$output_file"
