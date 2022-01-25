#date: 2022-01-25T16:54:36Z
#url: https://api.github.com/gists/ec4d53d581818a0ef9bf8cf15abd21b9
#owner: https://api.github.com/users/simon376

import yagmail
import tensorflow as tf
from tensorflow import keras


class EmailCallback(keras.callbacks.Callback):
    message_contents = ""
    yag: yagmail.SMTP
    
    def __init__(self, to: str="user@gmail.com", train_size=None, test_size=None) -> None:
        super().__init__()
        print("setting up yagmail...")
        # see yagmail documentation for setup. use username-password or oauth2.0 credentials
        self.yag = yagmail.SMTP("user@gmail.com", oauth2_file="~/oauth2_creds.json")
        self.to = to
        if train_size is not None:
            self.message_contents += f"size of train dataset: {train_size}\n"
        if test_size is not None:
            self.message_contents += f"size of test dataset: {test_size}\n"


    def on_train_begin(self, logs=None):
        msg = "Start Training with Model...\n"
        # msg += self.get_model_summary(self.model) # model summary already as plot attached
        keys = str(list(logs.keys()))
        msg += f"Starting training; got log keys: {keys}\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])
        self.message_contents += (msg + "\n\n---\n")
        print(msg)

    def on_train_end(self, logs=None):
        keys = str(list(logs.keys()))
        msg = f"Stop training; got log keys: {(keys)}\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])

        self.message_contents += (msg + "\n\n---\n")
        print(msg)

    def on_test_begin(self, logs=None):
        keys = str(list(logs.keys()))
        msg = f"Starting testing; got log keys: {keys}\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])
        self.message_contents += (msg + "\n\n---\n")
        print(msg)

    def on_test_end(self, logs=None):
        keys = str(list(logs.keys()))
        msg = f"stop testing; got log keys: {keys}\n"
        msg += str([f"{k}: {v}" for k,v in logs.items()])

        self.message_contents += (msg + "\n\n---\n")
        print(msg)
        print("sending e-mail..")
        self.send_message()

    def send_message(self):
        import datetime
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        import os
        import shutil

        # manually create & delete temp folder, since python tempfile doesn't seem to work
        res = ""
        temp_dir = "./temp/"
        os.mkdir(temp_dir)
        fn = os.path.join(temp_dir, "plot.png")
        # needs pydot & graphviz
        keras.utils.plot_model(
            self.model, 
            to_file=fn, 
            show_shapes=True, 
            rankdir="LR")
        
        res = self.yag.send(
            to=self.to, 
            subject=f"TensorFlow Training Callback {date}", 
            contents=self.message_contents,
            attachments=fn)
        shutil.rmtree(temp_dir)
        print("e-mail sent.")
        print(res)

    # method to parse model.summary() to string instead of console
    @staticmethod
    def get_model_summary(model):
        import io
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        return summary_string