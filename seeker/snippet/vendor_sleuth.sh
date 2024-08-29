#date: 2024-08-29T17:02:18Z
#url: https://api.github.com/gists/0cd89a25c730ac267559c44b5487c9ff
#owner: https://api.github.com/users/spezifisch

export DEVICE_BASE=$HOME/android/lineage/device/xiaomi/veux
export VENDOR_BASE=$HOME/android/lineage/vendor/xiaomi/veux/proprietary
export STOCK_BASE=$HOME/Dumps/veux-stock-vendor-20240829-1

for x in $(cat "$DEVICE_BASE/proprietary-files.txt" | cut -d'|' -f1 | grep -Ev '(^#|^$)'); do
	F="$VENDOR_BASE/$x"

	if [ -e "$F" ]; then
		#echo "found $x in vendor"
		
		G="$STOCK_BASE/$x"
		if [ -e "$G" ]; then
			#echo "found $x in stock"
			
			if diff -q "$F" "$G" > /dev/null; then
				# same files
				echo "match-vendor/stock $x"
			else
				echo "mismatch-vendor/stock $x"
			fi
		else
			echo "missing-compare $x"
		fi
	else
		echo "extraneous $x"
	fi
done