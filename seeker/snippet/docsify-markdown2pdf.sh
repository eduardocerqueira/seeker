#date: 2022-01-20T17:02:55Z
#url: https://api.github.com/gists/b16fc7967957d324f01adcb02a39e703
#owner: https://api.github.com/users/extratone

# Install deps
# sudo apt-get install -y texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra
# sudo apt-get install -y texlive-xetex
# sudo apt install -y ghostscript
# wget https://github.com/jgm/pandoc/releases/download/2.9.2.1/pandoc-2.9.2.1-1-amd64.deb
# sudo apt install -y ./pandoc-2.9.2.1-1-amd64.deb
# rm ./pandoc-2.9.2.1-1-amd64.deb

#cp fonts/*.ttf ~/.fonts # I use Google Fonts (Roboto -> please see line 8)
cd book # directory where markdown files can be found
alldirs=$(find . -type d | paste -sd:) # get all directories
#echo $alldirs
tmp=$(grep -oP '(?<=]\().*(?=\))' _sidebar.md | tr '\r\n' ' ') # get all markdown files in order
#echo $tmp
echo "Creating pdf..."
pandoc $tmp -o README.pdf "-fmarkdown-implicit_figures -o" --from=markdown -V geometry:margin=.6in --toc --toc-depth=1 --resource-path $alldirs --variable urlcolor=cyan --pdf-engine=xelatex --wrap=preserve -V documentclass=report -V 'mainfont:Roboto-Regular' -V 'mainfontoptions:BoldFont=Roboto-Bold, ItalicFont=Roboto-Italic, BoldItalicFont=Roboto-BoldItalic' --pdf-engine=xelatex
echo "Merging cover"
gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile="Documentation.pdf" ../cover.pdf README.pdf # to add a cover (merge)
rm README.pdf
echo "Finish."