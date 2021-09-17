#date: 2021-09-17T16:59:11Z
#url: https://api.github.com/gists/dd62bf958816690aa4ebdcd49b7909fb
#owner: https://api.github.com/users/MrJake222

async def json_dump_stream(obj, write):
    if obj is None:
        await write("null")

    elif isinstance(obj, str):
        await write('"')
        await write(obj)
        await write('"')

    elif isinstance(obj, bool):
        if obj:
            await write('true')
        else:
            await write('false')

    elif isinstance(obj, int) or isinstance(obj, float):
        await write(str(obj))

    elif isinstance(obj, bytes):
        # treating bytes as ascii string
        await write('"')
        for x in obj: await write(chr(x))
        await write('"')

    elif isinstance(obj, dict):
        await write('{')
        for i, key in enumerate(obj):
            await write('"')
            await write(key)
            await write('":')
            await json_dump_stream(obj[key], write)
            if i < len(obj)-1: await write(',')
        await write('}')

    elif isinstance(obj, list) or isinstance(obj, tuple):
        await write('[')
        for i, entry in enumerate(obj):
            await json_dump_stream(entry, write)
            if i < len(obj)-1: await write(',')
        await write(']')

    else:
        raise ValueError("unsupported value {}".format(type(obj)))

gc.collect()
f1 = gc.mem_free()
await json_dump_stream(self._body, stream_write)
gc.collect()
f2 = gc.mem_free()
logger.debug("took {} bytes of memory".format(f1-f2))
# outputs: took 0 bytes of memory