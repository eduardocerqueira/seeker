#date: 2023-02-20T16:45:56Z
#url: https://api.github.com/gists/6e3f07946a7ab6464b86e234bd7e0928
#owner: https://api.github.com/users/JamesTheAwesomeDude

def file_iterator(f, size=None, strict=False):
    if strict:
        return assertfilter(file_iterator(f, size), (lambda chunk, size=size: len(chunk) == size))
    while (chunk := f.read(size)):
        yield chunk

def infinite_iterator(iterator, restartfunc):
    while True:
        yield from iterator
        restartfunc(iterator)

def loop_file_chunkwise(f, size):
    yield from infinite_iterator(file_iterator(f, size), (lambda _, f=f: f.seek(0)))

def assertfilter(iterator, predicate):
    # TODO support send()
    for result in iterator:
        if not predicate(result):
            raise AssertionError("predicate failed in assertfilter()")
        yield result
