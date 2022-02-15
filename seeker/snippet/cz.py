#date: 2022-02-15T17:00:21Z
#url: https://api.github.com/gists/6f445047925aaaea6f9006f85366a3de
#owner: https://api.github.com/users/jn0

from binascii import hexlify, unhexlify
import json
import asn1

with open('oids.json') as f:
    oids = json.load(f)

with open('user.crt', 'rb') as f:
    data = f.read()

class XZ:
    def __init__(self, x, cls=None):
        self.value = x
        self.cls = cls
        self.name = '<' + (cls.__name__ if cls and hasattr(cls, '__name__') else '') + '#' + str(x) + '>'

def cast(t, v):
    try:
        return t(v)
    except ValueError:
        return XZ(v, cls=t)


OID = asn1.Tag(nr=6, typ=0, cls=0)
Universal_Primitive_OctetString = asn1.Tag(nr=4, typ=0, cls=0)
Context_Primitive_Boolean = asn1.Tag(nr=1, typ=0, cls=128)
Universal_Primitive_BitString = asn1.Tag(nr=3, typ=0, cls=0)
ForceDecode = (Universal_Primitive_OctetString, Universal_Primitive_BitString)

def oid(value):
    return 'OID.' + value
    return 'OID:' + '; '.join(filter(None, oids.get(value, [value])))

def string(tag, value):
    if tag == OID:
        return oid(value)
    if isinstance(value, str):
        return '('+str(len(value))+')'+value
    if isinstance(value, bytes):
        try:
            return string(tag, value.decode())
        except:
            return '('+str(len(value))+')'+hexlify(value).decode()
    if hasattr(value, '__iter__'):
        return '('+str(len(value))+')'+hexlify(value).decode()
    return value

def unlist1(r):
    return r[0] if len(r) == 1 else r

def obj(r):
    return {r[0]: obj(r[1:])} \
           if len(r) > 1 and isinstance(r[0], str) and r[0].startswith('OID:') and not (isinstance(r[1], str) and r[1].startswith('OID:')) \
           else unlist1(r)

def decode(data, indent=0):
    decoder = asn1.Decoder()
    decoder.start(data)

    r = []
    while not decoder.eof():
        tag, value = decoder.read()
        print(('  ' * indent), end='')
        print(str(tag) + ': ', end='')
        nr, typ, cls = cast(asn1.Numbers, tag.nr), cast(asn1.Types, tag.typ), cast(asn1.Classes, tag.cls)
        print(cls.name, typ.name, nr.name, ': ', end='')
        if typ == asn1.Types.Constructed or tag in ForceDecode:
            print('' if typ == asn1.Types.Constructed else '!>>')
            try:
                t = decode(value, indent=indent+1)
                r.append(t)
            except asn1.Error:
                print(('  ' * (indent + 1)), end='')
                s = string(tag, value)
                p = '(' + str(len(value)) + ')'
                x = hexlify(value).decode()
                print(s + ((' =' + x) if s != (p+x) else ''))
                r.append(string(tag, value))
        else:
            print(string(tag, value))
            r.append(string(tag, value))
            # print(('  ' * indent), '+', type(value), value)
    return obj(r)

r = decode(data)
pk = r[0][6]

print(json.dumps(pk, ensure_ascii=False, indent=2))
assert pk[0][0].startswith('OID.1.2.643.7.1.1.1.')
pk = unhexlify(pk[-1].split(')', 1)[-1])
print(pk[0])
assert pk[0] == 0
pk = decode(pk[1:])
print('===>', pk)
l, pk = pk.split(')', 1)
print('===>', pk, len(pk))
pk = unhexlify(pk)
d = len(pk) // 2
x, y = pk[:d][::-1], pk[d:][::-1]
print('X==>', hexlify(x).decode())
print('Y==>', hexlify(y).decode())

with open('user.crt.json', 'w') as f:
    json.dump(r, f, ensure_ascii=False, indent=2)
