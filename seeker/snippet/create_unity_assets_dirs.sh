#date: 2024-04-16T17:01:14Z
#url: https://api.github.com/gists/4326951aa7ee2a1ef17a366cff451394
#owner: https://api.github.com/users/uchidama

#!/bin/bash
#
# Unityフォルダ構成のルールについて
# https://qiita.com/takish/items/8608ba9070755da3ae6d
#  ここで書かれたフォルダ構成をUnityの空のプロジェクトを作ったあとに作成するスクリプト
#  空のプロジェクトのルートフォルダにおいて実行

# Base directory for Unity project Assets
base_dir="./Assets"

# List of directories to create
directories=(
    "Scenes"
    "Prefabs"
    "Scripts"
    "Animations"
    "Materials"
    "PhysicsMaterials"
    "Fonts"
    "Textures"
    "Audio"
    "Resources"
    "Editor"
    "Plugins"
)

# Create each directory
for dir in "${directories[@]}"; do
    mkdir -p "$base_dir/$dir"
done

echo "Directory structure created successfully in $base_dir."


