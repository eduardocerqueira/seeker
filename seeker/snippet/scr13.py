#date: 2022-02-15T16:57:47Z
#url: https://api.github.com/gists/aeaf777a6d2d8be93445f44af0874ce8
#owner: https://api.github.com/users/vaibhavtmnit

sampler = torch.utils.data.RandomSampler
testdataloder = DataLoader(testdataset, batch_size = 4, shuffle=False)

for dl in testdataloder:
    
    print(dl[0])

"""

tensor([[-1.1258, -1.1524, -0.2506, -0.4339],
        [ 0.8487,  0.6920, -0.3160, -2.1152],
        [ 0.3223, -1.2633,  0.3500,  0.3081],
        [ 0.1198,  1.2377,  1.1168, -0.2473]])
tensor([[-1.3527, -1.6959,  0.5667,  0.7935],
        [ 0.5988, -1.5551, -0.3414,  1.8530],
        [-0.2159, -0.7425,  0.5627,  0.2596],
        [-0.1740, -0.6787,  0.9383,  0.4889]])
tensor([[ 1.2032,  0.0845, -1.2001, -0.0048],
        [-0.5181, -0.3067, -1.5810,  1.7066]])

"""


sampler = torch.utils.data.RandomSampler(testdataset)
# Dataloader with samples
# Please note: Sampler is mutually exclusinve to shuffle argument
testdataloder_v2 = DataLoader(testdataset, batch_size = 4,sampler=sampler)

for dl in testdataloder_v2:

    
    print(dl[0])
    
"""
tensor([[ 0.1198,  1.2377,  1.1168, -0.2473],
        [ 0.3223, -1.2633,  0.3500,  0.3081],
        [-1.1258, -1.1524, -0.2506, -0.4339],
        [-0.2159, -0.7425,  0.5627,  0.2596]])
tensor([[ 0.8487,  0.6920, -0.3160, -2.1152],
        [ 0.5988, -1.5551, -0.3414,  1.8530],
        [-0.5181, -0.3067, -1.5810,  1.7066],
        [ 1.2032,  0.0845, -1.2001, -0.0048]])
tensor([[-0.1740, -0.6787,  0.9383,  0.4889],
        [-1.3527, -1.6959,  0.5667,  0.7935]])

"""