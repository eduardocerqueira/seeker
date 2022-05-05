#date: 2022-05-05T16:57:44Z
#url: https://api.github.com/gists/7d1cf10d8a80a60971629b417b112e0d
#owner: https://api.github.com/users/sukhitashvili

import pickle


class CustomUnpickler(pickle.Unpickler):
    """
    Needs python3+
    Solves error with pickle load when it cannot load a pickled class.
    Usage:
        pickle_data = CustomUnpickler(open('file_path.pkl', 'rb')).load()
        OR:
            with open(model_name, "rb") as file:
                pickle_data = CustomUnpickler(file).load()
    """
    def find_class(self, module, name):
        if name == 'StackpoleDetector':
            from src.models.stackpole_detector import StackpoleDetector
            return StackpoleDetector
        return super().find_class(module, name)