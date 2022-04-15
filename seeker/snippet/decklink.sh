#date: 2022-04-15T16:53:41Z
#url: https://api.github.com/gists/6aa1bdd9622b3d72b1936855b95b705c
#owner: https://api.github.com/users/Manouchehri

ffmpeg -y -raw_format yuv422p10 -timecode_format rp188any -video_input hdmi -f decklink -i 'UltraStudio Recorder 3G' -c:a aac_at -c:v hevc_videotoolbox -profile:v main10 -b:v 10M -aac_at_mode cbr -b:a 256K -tag:v hvc1 output.mov
