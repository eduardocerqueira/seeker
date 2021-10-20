#date: 2021-10-20T16:56:49Z
#url: https://api.github.com/gists/19aaaec337482f9db9adbe449c2fd9ca
#owner: https://api.github.com/users/chirsz-ever

#!/usr/bin/env bash
cat <<EOF
use std::io;
use std::io::prelude::*;
fn main() -> io::Result<()> {
    print!("请给出一个不多于5位的正整数:");
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let x = line.trim().parse::<i32>().expect("not a i32");
    match x {
EOF

indent="    "
indent2="${indent}${indent}"
indent3="${indent2}${indent}"
units=('个' '十' '百' '千' '万')

for ((x=1;x<=99999;++x)); do
	xlen=${#x}
	rx=""
	echo "${indent2}$x => {"
	echo "${indent3}println!(\"是${xlen}位数\");"
	for ((i=0;i<xlen;++i)); do
		k=${x:$((xlen-i-1)):1}
		echo "${indent3}println!(\"${units[$i]}位数是：$k\");"
		rx=$k$rx
	done
	echo "${indent3}println!(\"倒过来是：$rx\");"
	echo "${indent2}}"
done
echo "${indent2}_ => {}"
echo "${indent}}"
echo "${indent}Ok(())"
echo "}"
