#date: 2023-01-30T16:54:51Z
#url: https://api.github.com/gists/69203b58c5b5b4dfe8ebf7f1c43ed517
#owner: https://api.github.com/users/malizia-g

#pip install Image
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras

model = keras.models.load_model('model.h5')

# Caricamento dell'immagine
img = Image.open("8.jpg")

# Riduzione della risoluzione dell'immagine
img = img.resize((28, 28))

#Converto in scala di grigi
img = img.convert('L')

# Normalizzazione dei valori dei pixel
img = np.array(img) / 255.0

# Trasformazione dell'immagine in un tensore
img = img[tf.newaxis, ..., tf.newaxis]

# Previsione del modello
predictions = model.predict(img)

# Calcolo della classe con la probabilità più alta
predicted_class = tf.argmax(predictions[0]).numpy()

print(predictions)

print(predicted_class)
