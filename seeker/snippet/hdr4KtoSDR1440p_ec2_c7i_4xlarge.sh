#date: 2025-12-30T16:58:26Z
#url: https://api.github.com/gists/a04b8316ab52d17c13457ad7deb0d997
#owner: https://api.github.com/users/Trogious

#!/bin/sh -

for f in `ls Prefix*.mkv`
do
	OUT=`echo -n $f | sed -nE 's/.+(S[[:digit:]][[:digit:]]E[[:digit:]][[:digit:]]).+\.mkv/\1/p'`
	ffmpeg -i "$f" \
	  -map 0 \
	  -map -0:t \
	  -map_metadata 0 \
	  -map_chapters 0 \
	  -vf "zscale=w=2560:h=1440:t=linear:npl=100,
	       format=gbrpf32le,
	       zscale=p=bt709,
	       tonemap=tonemap=hable:desat=0,
	       zscale=t=bt709:m=bt709:r=tv,
	       format=yuv420p10le" \
	  -c:v libx265 \
	  -crf 16 \
	  -preset slow \
	  -pix_fmt yuv420p10le \
	  -c:a copy -sn \
	  -threads 14 -filter_threads 4 \
	  ${OUT}.mkv > ${OUT}.log 2>&1
	  /home/ec2-user/notify.sh "$?" "$OUT"
done
sleep 10 && sudo /sbin/poweroff
# -benchmark -benchmark_all \
# sudo nice -n -20 ./hdr4KtoSDR1440p_ec2_c7i_4xlarge.sh