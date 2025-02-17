#date: 2025-02-17T17:12:26Z
#url: https://api.github.com/gists/e026133d02ce6391ff1c4f9c5ab044e3
#owner: https://api.github.com/users/mi3law

import ao_arch as ar

description = "Basic Recommender System"

#genre, length
arch_i = [5, 2, 3, 1, 2]   # genre_encoding, age_category, mood, ana (animation or not), satisfaction_level
arch_z = [1]           
arch_c = []           
connector_function = "full_conn"

# To maintain compatibility with our API, do not change the variable name "Arch" or the constructor class "ao.Arch" in the line below (the API is pre-loaded with a version of the Arch class in this repo's main branch, hence "ao.Arch")

Arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, description)
