#date: 2021-12-16T17:04:15Z
#url: https://api.github.com/gists/1312d07f56ca5175a369c8e13975ec6b
#owner: https://api.github.com/users/mapmeld

# All I'm looking for on an ML example:
# ! pip install name_of_library

from name_of_library import model, other_stuff

tdata = load_data_from_file() # not a built-in datasets source where I'd need to write python to add data
tdata.apply(changes) # whose dataset is so perfect we don't edit it

model.train(tdata, **explained_params)

real_output = model.predict(real_input)