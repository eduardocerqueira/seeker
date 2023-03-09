#date: 2023-03-09T16:52:37Z
#url: https://api.github.com/gists/e967fc5b3ad9d146406d3d6e3b53828f
#owner: https://api.github.com/users/Noe-Sanchez

makefile_contents='project.analyze:\n\t./.handler.sh analyze\n\techo "- Done analyzing"\nproject.integrate:\n\t./.handler.sh integrate\n\techo "- Consolidated top entity"\nproject.clean:\n\tghdl --clean\n\techo "- Cleaned stack"\nwave.gen:\n\techo "- Dumping signals"\n\t./.handler.sh run\nwave.show:\n\techo "- Opening wave"\n\t./.handler.sh open\nproject.all: project.clean project.analyze project.integrate wave.gen wave.show'

touch ./Makefile
echo $makefile_contents > ./Makefile

wget https://gist.githubusercontent.com/Noe-Sanchez/c2b16bc110bb30555ad87be2c24c07c2/raw/64bba8a6a5a8e52962440e48002cfc4faef4c24c/handler.sh
mv ./handler.sh ./.handler.sh
chmod +x ./.handler.sh

rm "$0"