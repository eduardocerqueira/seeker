#date: 2025-03-07T16:49:40Z
#url: https://api.github.com/gists/9bbbc6cec404bd6f9a4b089ede332817
#owner: https://api.github.com/users/nithyadurai87

from tensorflow.keras.models import model_from_json
import pickle

tokens = "**********"
model_file = pickle.load(open(r'/content/Ilayaraja_book_model.pkl', 'rb'))

model = model_from_json(model_file['model_json'])
model.set_weights(model_file['model_weights'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

input_text = "பெரும் மேதைகளை"
predict_next_words= 3

for _ in range(predict_next_words):
    input_token = "**********"
    input_x = "**********"=max_line_len-1, padding='pre')
    predicted = np.argmax(model.predict(input_x), axis=-1) 
    output_word = ""
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"w "**********"o "**********"r "**********"d "**********", "**********"  "**********"i "**********"n "**********"d "**********"e "**********"x "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********". "**********"w "**********"o "**********"r "**********"d "**********"_ "**********"i "**********"n "**********"d "**********"e "**********"x "**********". "**********"i "**********"t "**********"e "**********"m "**********"s "**********"( "**********") "**********": "**********"
        if index == predicted:
            output_word = word
            break
    input_text += " " + output_word

print(input_text)