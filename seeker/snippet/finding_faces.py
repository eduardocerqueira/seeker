#date: 2022-04-21T17:07:58Z
#url: https://api.github.com/gists/08e55621e2b168c0c260ab4e81175ab3
#owner: https://api.github.com/users/diegounzueta

# load the cv2 face model
face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# dictionary to store information about the location faces in each image
find_face = {
            "file":[],
            "x":[],
            "y":[],   
            "w":[],
            "h":[],
            }

for dirname, _, filenames in os.walk('./images/'):
    # for each image
    for filename in filenames:
        #read the image
        img = cv2.imread("./images/" + filename)
        img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
        
        #detect faces using face model (gives back x,y,w,h)
        faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4) #returns a list of (x,y,w,h) tuples
        
        if len(faces) == 0:
            faces = [[0,0,0,0]]
        #for each face 
        for square in faces:
            find_face["file"] += [filename[:-4]]
            find_face["x"] += [square[0]]
            find_face["y"] += [square[1]]
            find_face["w"] += [square[2]]
            find_face["h"] += [square[3]]
            
df_find_face = pd.DataFrame(find_face)