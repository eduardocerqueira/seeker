#date: 2024-09-09T16:59:52Z
#url: https://api.github.com/gists/7243c30e2b967421b41a4d57ba17463d
#owner: https://api.github.com/users/hardikfuria12

from kivy import platform
from jnius import autoclass
if platform == "android":
    DataSnapshot = autoclass('com.google.firebase.database.DataSnapshot')
    HashMap = autoclass('java.util.HashMap')
    ArrayList = autoclass('java.util.ArrayList')
    Boolean = autoclass('java.lang.Boolean')
    String = autoclass('java.lang.String')
    Integer = autoclass('java.lang.Integer')
    Double = autoclass('java.lang.Double')

class BaseScreenController:
    def __init__(self, app, model):
        self.app = app
        self.model = model
        self.view = None
        
    def set_view(self, view):
        self.view = view
        
    def get_view(self):
        return self.view
    
    def convert_data_snapshot_to_dict(self,data_snapshot):
        def data_snapshot_to_dict(snapshot):
            result = {}
            for child in snapshot.getChildren():
                key = child.getKey()
                value = child.getValue()
                if isinstance(value, DataSnapshot):
                    result[key] = data_snapshot_to_dict(value)
                elif isinstance(value, HashMap):
                    result[key] = dict_from_hash_map(value)
                elif isinstance(value, ArrayList):
                    result[key] = list_from_array_list(value)
                else:
                    result[key] = value
            return result

        def dict_from_hash_map(hash_map):
            result = {}
            for entry in hash_map.entrySet():
                key = str(entry.getKey())
                value = entry.getValue()
                if isinstance(value, HashMap):
                    result[key] = dict_from_hash_map(value)
                elif isinstance(value, ArrayList):
                    result[key] = list_from_array_list(value)
                else:
                    result[key] = value
            return result

        def list_from_array_list(array_list):
            result = []
            for item in array_list:
                if isinstance(item, HashMap):
                    result.append(dict_from_hash_map(item))
                elif isinstance(item, ArrayList):
                    result.append(list_from_array_list(item))
                else:
                    result.append(item)
            return result

        return data_snapshot_to_dict(data_snapshot)
    
    def convert_dict_to_java_obj(self,py_dict):
        java_map = HashMap()
    
        for key, value in py_dict.items():
            java_key = String(str(key))
            
            if isinstance(value, dict):
                java_value = self.convert_dict_to_java_obj(value)
            elif isinstance(value, list):
                java_list = ArrayList()
                for item in value:
                    if isinstance(item, dict):
                        java_item = self.convert_dict_to_java_obj(item)
                    else:
                        java_item = String(str(item))
                    java_list.add(java_item)
                java_value = java_list
            else:
                java_value = String(str(value))
            
            java_map.put(java_key, java_value)
        
        return java_map
