#date: 2023-01-10T17:06:19Z
#url: https://api.github.com/gists/ca6deae7dd5332100f443457845f38f1
#owner: https://api.github.com/users/xtrmstep

# source: https://stackoverflow.com/a/50156142/2833774

def flatten(schema, prefix=None):
    fields = []
    for field in schema.fields:
        name = prefix + '.' + field.name if prefix else field.name
        alias_name = name.replace(".", "__")
        dtype = field.dataType
        if isinstance(dtype, pst.ArrayType):
            dtype = dtype.elementType

        if isinstance(dtype, pst.StructType):
            fields += flatten(dtype, prefix=name)
        else:
            fields.append(col(name).alias(alias_name))

    return fields