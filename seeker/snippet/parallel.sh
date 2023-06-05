#date: 2023-06-05T17:01:30Z
#url: https://api.github.com/gists/58dc3dcfee808f1b30492325a3a85f3b
#owner: https://api.github.com/users/ezequieljsosa

sudo apt install parallel

wget https://files.wwpdb.org/pub/pdb/derived_data/index/entries.idx

cut -f1 entries.idx  | grep -v "-" | awk '{print tolower($pdb)}' > pdbs.lst
cat pdbs.lst | parallel -j3 'export pdb={};idx=${pdb:1:2};  wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 10 -q  https://files.wwpdb.org/pub/pdb/data/structures/divided/pdb/${idx}/pdb${pdb}.ent.gz -O pdbs/${pdb}.pdb.gz'