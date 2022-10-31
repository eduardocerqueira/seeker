#date: 2022-10-31T17:14:28Z
#url: https://api.github.com/gists/f3e8877302b0fe3227dc17b86ebd3866
#owner: https://api.github.com/users/leesw1347

def hash_file(file, block_size=65536):
    hasher = hashlib.md5()
    for buf in iter(partial(file.read, block_size), b''):
        hasher.update(buf)

    return hasher.hexdigest()


def upload_to(instance, filename):
    """
    :type instance: dolphin.models.File
    """
    instance.file.open()
    filename_base, filename_ext = os.path.splitext(filename)

    return "{0}.{1}".format(hash_file(instance.file), filename_ext)
