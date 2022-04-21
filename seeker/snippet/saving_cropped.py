#date: 2022-04-21T17:21:39Z
#url: https://api.github.com/gists/977894a66ab969783438a9eeb95c6608
#owner: https://api.github.com/users/diegounzueta

def save_faces(df):
    #for each face in dataframe
    for index, face in df.iterrows():
        #open image
        filename = "./images/" + face["file"].split("\\")[1] + ".png"
        im = Image.open(filename)
        #crop image
        im = im.crop( face[["xmin", "ymin", "xmax", "ymax"]].values)
        #save image
        save_name = ".//processed//" + face["name"] + "//" + face["file"] + "_" + str(index) + ".png"
        im.save(save_name)