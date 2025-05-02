#date: 2025-05-02T17:02:09Z
#url: https://api.github.com/gists/9ee32703dc3570e2ca3428fb6278a7a8
#owner: https://api.github.com/users/xanathar

# You can get the profile id with flatpak run --command=gsettings app.devsuite.Ptyxis get org.gnome.Ptyxis default-profile-uuid

export uuid=<PASTE YOUR PROFILE ID HERE>
flatpak run --command=gsettings app.devsuite.Ptyxis set org.gnome.Ptyxis.Profile:/org/gnome/Ptyxis/Profiles/${uuid}/ opacity 0.85