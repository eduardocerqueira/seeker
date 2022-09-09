#date: 2022-09-09T17:04:28Z
#url: https://api.github.com/gists/6a247081b4d541057d2b7c7354609a46
#owner: https://api.github.com/users/senthilnayagam

echo "count: $1";
echo "prompt: $2";
echo "output_file_basename: $3";
echo "prompt: $2" > prompt_$3.txt;
for i in $(seq 1 1 $1)
do
echo "python3.9 demo.py --prompt \"$2\" --output output_$3_$i.png";
   python3.9 demo.py --prompt "$2" --output output_$3_$i.png
done