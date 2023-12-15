#date: 2023-12-15T16:37:22Z
#url: https://api.github.com/gists/05bbe5aa80a096c15cd3cb789cc1007f
#owner: https://api.github.com/users/mckirk

from collections import Counter
from functools import reduce
import re


def camel_to_snake(name):
    """
    Really naive camelCase to snake_case conversion
    Note: turns e.g. someID into some_i_d (on purpose, to ensure the inverse process returns the original name)
    """
    return re.sub(r'([A-Z])', r'_\1', name).lower()


class FieldType:
    def __init__(self, can_be: set = None, child_types: dict = None, list_elem_type: "FieldType" = None):
        self.can_be = can_be or set()
        self.child_types = child_types
        self.list_elem_type = list_elem_type

    def __bool__(self):
        return True

    def merge(self, other: "FieldType"):
        new_type = FieldType()
        new_type.can_be = self.can_be | other.can_be

        if self.child_types and other.child_types:
            merged = dict()
            for k in (self.child_types.keys() | other.child_types.keys()):
                self_type = self.child_types.get(k, FieldType(can_be={'missing'}))
                other_type = other.child_types.get(k, FieldType(can_be={'missing'}))
                merged[k] = self_type.merge(other_type)
            new_type.child_types = merged
        else:
            new_type.child_types = self.child_types or other.child_types

        if self.list_elem_type and other.list_elem_type:
            new_type.list_elem_type = self.list_elem_type.merge(other.list_elem_type)
        else:
            new_type.list_elem_type = self.list_elem_type or other.list_elem_type

        return new_type

    @classmethod
    def merge_list(cls, type_list):
        return reduce(FieldType.merge, type_list, FieldType())

    @classmethod
    def get_type(cls, obj):
        if obj is None:
            return cls(can_be={'none'})

        if isinstance(obj, bool):
            return cls(can_be={'bool'})

        if isinstance(obj, int):
            return cls(can_be={'int'})
        
        if isinstance(obj, float):
            return cls(can_be={'float'})

        if isinstance(obj, str):
            return cls(can_be={'str'})

        if isinstance(obj, list):
            return cls(
                can_be={'list'},
                list_elem_type=cls.merge_list([cls.get_type(o) for o in obj]))

        if isinstance(obj, dict):
            return cls(
                can_be={'dict'},
                child_types={k: cls.get_type(v) for k, v in obj.items()}
            )

        assert False

    def to_python_type(self, class_builder: "ClassBuilder", field_name: str):
        pos_types = []
        for t in ['bool', 'int', 'str', 'float']:
            if t in self.can_be:
                pos_types.append(t)
        if 'list' in self.can_be:
            assert self.list_elem_type is not None
            assert 'missing' not in self.list_elem_type.can_be
            pos_types.append(f"list[{self.list_elem_type.to_python_type(class_builder, field_name.rstrip('s'))}]")
        if 'dict' in self.can_be:
            assert self.child_types is not None
            pos_types.append(class_builder.build(field_name, self.child_types))
        if 'none' in self.can_be or 'missing' in self.can_be:
            pos_types.append('None')

        t = " | ".join(pos_types)

        if 'missing' in self.can_be:
            t += " = None"

        return t


class ClassBuilder:
    def __init__(self, decamelcase: bool):
        self.class_name_count = Counter()
        self.classes = dict()
        self.decamelcase = decamelcase

    def build(self, name: str, child_types: dict[str, FieldType]):
        class_name_idx = self.class_name_count[name]
        class_name = name.capitalize()
        if class_name_idx:
            class_name += str(class_name_idx)
        self.class_name_count[name] += 1

        mandatory_fields = []
        optional_fields = []
        for k, t in child_types.items():
            k_name = camel_to_snake(k) if self.decamelcase else k
            line = f"    {k_name}: {t.to_python_type(self, k)}"
            if 'missing' in t.can_be:
                optional_fields.append(line)
            else:
                mandatory_fields.append(line)

        mandatory_fields.sort()
        optional_fields.sort()

        if optional_fields:
            optional_fields = ["    # Optional"] + optional_fields

        model = "CamelCaseModel" if self.decamelcase else "BaseModel"
        lines = [f"class {class_name}({model}):"] + mandatory_fields + optional_fields

        self.classes[class_name] = "\n".join(lines)

        return class_name

    def dump_classes(self):
        return "\n\n\n".join(self.classes.values())


def get_classes(json):
    root_type = FieldType.get_type(dict(root=json))
    class_builder = ClassBuilder(decamelcase=True)
    root_type.to_python_type(class_builder, 'response')
    return class_builder.dump_classes()