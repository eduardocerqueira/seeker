#date: 2022-12-21T16:36:00Z
#url: https://api.github.com/gists/9ee9d40d27c1c62d9e13e75cb8936876
#owner: https://api.github.com/users/AureumApes

sed -i '/brainfuck/d' $HOME/.zshrc
sed -i '/.brainfuck/d' $HOME/.zshrc

echo "# Add brainfuck to Path" >> $HOME/.zshrc
echo "export PATH=\"\$PATH:$HOME/.brainfuck/bin\"" >> $HOME/.zshrc

bin_dir="$HOME/.brainfuck/bin/"
if [ -d $bin_dir ]; then
  echo ""
else
  mkdir $HOME/.brainfuck
  mkdir $HOME/.brainfuck/bin
fi

cd $HOME/.brainfuck/bin/
wget -O brainfuck https://github.com/AureumApes/BrainFuckInterpreter/releases/download/v1.0/bf-interpreter
chmod +x brainfuck