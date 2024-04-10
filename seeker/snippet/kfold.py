#date: 2024-04-10T16:52:57Z
#url: https://api.github.com/gists/ed618033f6a641ddc8368de932ef1ed6
#owner: https://api.github.com/users/jamesdolezal

dataset = P.dataset(299, 302, filters={'type': ['primary', 'node']})

k1, k2, k3 = sf.util.get_splits('splits.json', id=0)

for train_slides, val_slides in (((k2, k3), k1)
                                 ((k1, k3), k2),
                                 ((k1, k2), k3)):

    train_dts = dataset.filter({'slide': train_slides})    
    val_dts = dataset.filter({'slide': val_slides[0] + val_slides[1]})