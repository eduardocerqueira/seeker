#date: 2021-09-28T17:06:22Z
#url: https://api.github.com/gists/d14ed1c7b2df8b4ce1fc94d6ed4c9c9b
#owner: https://api.github.com/users/Achleshwar

torchreid.data.register_video_dataset('pigs', MyReidDataset)

datamanager = torchreid.data.VideoDataManager(
    root=ROOT_DIR,
    sources='pigs',
    transforms=None,
    workers=2,
    height=256,
    width=256
)


######################################
# You will get the following details as output
# Building train transforms ...
# + resize to 256x256
# + to torch tensor of range [0, 1]
# + normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Building test transforms ...
# + resize to 256x256
# + to torch tensor of range [0, 1]
# + normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# => Loading train (source) dataset
# => Loaded Pigs
#   -------------------------------------------
#   subset   | # ids | # tracklets | # cameras
#   -------------------------------------------
#   train    |     3 |        1227 |         4
#   query    |     3 |          48 |         4
#   gallery  |     3 |        1255 |         4
#   -------------------------------------------


#   **************** Summary ****************
#   source             : ['pigs']
#   # source datasets  : 1
#   # source ids       : 3
#   # source tracklets : 1227
#   # source cameras   : 4
#   target             : ['pigs']
#   *****************************************