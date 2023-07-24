#date: 2023-07-24T16:49:03Z
#url: https://api.github.com/gists/df0991bd7fb70b3b4488ee37b62f6f77
#owner: https://api.github.com/users/KRC2000

# get current theme
theme=$(kreadconfig5 --file ~/.config/kcminputrc --group Mouse --key cursorTheme)
# get ids of installed flatpak apps
app_ids=$(flatpak list --app | awk '{print $2}')

for id in $app_ids
do
    # one of these should fix the cursor theme
    flatpak --user override $id --filesystem=/home/$USER/.icons/:ro
    flatpak override --user --env=XCURSOR_THEME=$theme $id
done