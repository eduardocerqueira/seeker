#date: 2023-07-04T17:07:08Z
#url: https://api.github.com/gists/4672b60994bfadbfd8046544258005f1
#owner: https://api.github.com/users/stjhimy

# tested on macOS 10.12.4
# based on https://elixirforum.com/t/how-to-change-a-phoenix-project-name-smoothly/1217/6

# replace values as necessary
current_otp="hello_phoenix" 
current_name="HelloPhoenix"
new_otp=""
new_name=""

git grep -l $current_otp | xargs sed -i '' -e 's/'$current_otp'/'$new_otp'/g'
git grep -l $current_name | xargs sed -i '' -e 's/'$current_name'/'$new_name'/g'
mv ./lib/$current_otp ./lib/$new_otp
mv ./lib/$current_otp.ex ./lib/$new_otp.ex