#date: 2021-10-08T16:57:17Z
#url: https://api.github.com/gists/421641043e7cb3d367c401a2c4d8cb87
#owner: https://api.github.com/users/jon-betts

from copy import deepcopy


class DataMutator:
    @staticmethod
    def vary_simple(data=...):
        for mutated in (True, False, "string", 0, 123, 12.0, None):
            if data is ... or not isinstance(data, type(mutated)):
                yield mutated

    @classmethod
    def vary_complex(cls, data=...):
        yield from cls.vary_simple(data)

        for mutated in ({}, {"data": 1}, [], ["item"]):
            if data is ... or not isinstance(data, type(mutated)):
                yield mutated

    @staticmethod
    def vary_int(data):
        if data:
            yield -1 * data
            yield 0
            yield data * 1293762398784234
        else:
            yield 123456

    @staticmethod
    def vary_string(string):
        yield string * 1000
        if string:
            yield ""


class DictMutator:
    @staticmethod
    def pop_key(data):
        for key in data.keys():
            copied = deepcopy(data)
            copied.pop(key)
            yield copied

    @staticmethod
    def insert_key(data):
        copied = deepcopy(data)
        key = "__INSERTED_KEY__"
        while key in data:
            key += "_"

        for value in DataMutator.vary_complex():
            copied[key] = value
            yield copied

    @staticmethod
    def recursive(data):
        for key, value in data.items():
            copied = deepcopy(data)
            for mutation in mutate(value):
                copied[key] = mutation

                yield copied


class ListMutator:
    @staticmethod
    def pop_item(data):
        for pos, _ in enumerate(data):
            copied = deepcopy(data)
            copied.pop(pos)
            yield copied

    @staticmethod
    def append_item(data):
        copied = deepcopy(data)
        copied.append(None)
        for random in DataMutator.vary_complex():
            copied[-1] = random
            yield copied

    @staticmethod
    def recursive(data):
        for pos, item in enumerate(data):
            copied = deepcopy(data)
            for mutated_item in mutate(item):
                copied[pos] = mutated_item
                yield copied


MUTATORS = {
    list: (ListMutator.pop_item, ListMutator.append_item, ListMutator.recursive),
    dict: (DictMutator.pop_key, DictMutator.insert_key, DictMutator.recursive),
    str: (DataMutator.vary_string,),
    int: (DataMutator.vary_int,),
    (int, float, str, bool, list, dict): (DataMutator.vary_complex,),
}


def mutate(data):
    for mutator_type, mutators in MUTATORS.items():
        if isinstance(data, mutator_type):
            for mutator in mutators:
                yield from mutator(data)
