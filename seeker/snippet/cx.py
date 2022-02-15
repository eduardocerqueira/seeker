#date: 2022-02-15T17:00:21Z
#url: https://api.github.com/gists/6f445047925aaaea6f9006f85366a3de
#owner: https://api.github.com/users/jn0

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat._der import DERReader, SEQUENCE
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends.openssl.backend import hashes
from binascii import hexlify
import sys
import json

with open('oids.json', 'rt') as f:
    N_DECODER = {
        k: '; '.join(s for s in v if s)
        for k, v in json.load(f).items()
        if not k.startswith('_')
    }

N_DECODER.update({
    # https://oid.iitrust.ru
    '2.5.29.16': 'privateKeyUsagePeriod',  # https://oidref.com/2.5.29.16
})

for k in N_DECODER:
    try:
        o = x509.ObjectIdentifier(k)
    except:
        print('Bad OID', repr(k))
        sys.exit(1)
    if o not in x509.oid._OID_NAMES:
        x509.oid._OID_NAMES[o] = N_DECODER[k]
    if o not in x509._SIG_OIDS_TO_HASH:
        x509._SIG_OIDS_TO_HASH[o] = None

with open(sys.argv[1], 'rb') as f:
    cert = f.read()
cert_info = x509.load_der_x509_certificate(cert, default_backend())

#for i in dir(cert_info):
#    print(i) # ,':', getattr(cert_info, i))
#sys.exit()


def is_unknown(o): return o.oid._name == 'Unknown OID'


def read_2_5_29_16(data):
    o = DERReader(data.value.value).read_single_element(SEQUENCE)
    parsed = []
    while not o.is_empty():
        x = o.read_any_element()
        parsed.append(bytes(x[1].data).decode())
    return parsed


def read_1_2_643_100_111(data):
    o = DERReader(data.value.value).read_any_element()
    return bytes(o[1].data).decode()


def read_1_2_643_100_112(data):
    o = DERReader(data.value.value).read_any_element()
    value = bytes(o[1].data)
    r = []
    while value:
        b, value = value[0], value[1:]
        assert b == 12
        l, value = value[0], value[1:]
        s, value = value[:l], value[l:]
        r.append(s.decode())
    return r


V_DECODER = {
    '1.3.6.1.5.5.7.3.2':
        lambda x:
            f'{N_DECODER.get(x.dotted_string, x.dotted_string)}={x._name}',
    '1.3.6.1.5.5.7.3.4':
        lambda x:
            f'{N_DECODER.get(x.dotted_string, x.dotted_string)}={x._name}',
    '1.2.643.100.111': read_1_2_643_100_111,
    '1.2.643.100.112': read_1_2_643_100_112,
    '2.5.29.16': read_2_5_29_16,
}


def dump(t, x):
    print('###', t)

    def dump16(o, x):
        xx = ' '.join(f'{i:02x}' for i in x)
        cc = ''.join(chr(i) if chr(i).isprintable() else '.' for i in x)
        print(f'{o:08x} {xx:48s}|{cc:16s}|')
        return len(x)
    o, ll = 0, len(x)
    while x:
        o += dump16(o, x[:16])
        x = x[16:]
    print(f'{ll:08x} = {ll}.')


def name(x):
    return N_DECODER.get(x.oid.dotted_string, x.oid.dotted_string) \
           if is_unknown(x) else x.oid._name


def getKeyUsage(v):
    assert isinstance(v, x509.KeyUsage)
    r = []
    for f in ('content_commitment', 'crl_sign', 'data_encipherment',
              'decipher_only', 'digital_signature', 'encipher_only',
              'key_agreement', 'key_cert_sign', 'key_encipherment'):
        try:
            z = getattr(v, f)
        except ValueError:
            z = None
        if z:
            r.append(f)
    return '; '.join(r)


def uxv(x, v):
    o, v = x.oid, x.value
    f = V_DECODER.get(o.dotted_string)
    if f and callable(f):
        return str(f(x))
    return 'OID:%s VALUE:%s' % (o, v)


def xvalue(x):
    if not hasattr(x, 'value'):
        if hasattr(x, 'dotted_string'):
            return V_DECODER.get(x.dotted_string,
                                 lambda p: f'No-Value:{p!r}')(x)
        else:
            if isinstance(x, x509.AccessDescription):
                return f'{x.access_method._name!r}:{x.access_location.value!r}'
            if isinstance(x, x509.DistributionPoint):
                return f'DistributionPoint:{", ".join(i.value for i in x.full_name)}'  # noqa:E501
            if isinstance(x, x509.PolicyInformation):
                n = N_DECODER.get(x.policy_identifier.dotted_string,
                                  x.policy_identifier.dotted_string)
                return f'<{n}:{repr(x.policy_qualifiers) if x.policy_qualifiers else "-"}>'  # noqa:E501
            return f'NO-VALUE:{x!r}'
    v = x.value
    if hasattr(v, '__iter__'):
        return '; '.join(str(xvalue(i)) for i in v)
    if isinstance(v, x509.SubjectKeyIdentifier):
        return ' '.join(f'{i:02x}' for i in v.digest)
    if isinstance(v, x509.KeyUsage):
        return getKeyUsage(v)
    if isinstance(v, x509.AuthorityKeyIdentifier):
        return f'AuthorityKeyIdentifier:key_identifier={hexlify(v.key_identifier)}; authority_cert_serial_number={hex(v.authority_cert_serial_number)}; authority_cert_issuer={",".join(str(i.value) for i in v.authority_cert_issuer)}'  # noqa:E501
    if is_unknown(x):
        if x.oid.dotted_string in V_DECODER:
            return V_DECODER[x.oid.dotted_string](x)
        return x.value.value.decode('utf-8', 'replace')
    return uxv(x, v)


def show(pfx, obj, val):
    for a in obj:
        print(pfx+'.'+name(a), ':', val(a), end='\n\n')


class XZ:
    def __init__(self, oid): self.oid = oid

def mval(v):
    if isinstance(v, int):
        return str(v) + ' = ' + hex(v)
    return str(v)

for x in ['not_valid_after', 'not_valid_before', 'serial_number', 'version']:
    print(x, ':', mval(getattr(cert_info, x)))

print('signature_algorithm_oid:', name(XZ(cert_info.signature_algorithm_oid)))
print('signature:', hexlify(cert_info.signature).decode())

# these are ok, just way too long
# print('tbs_certificate_bytes:', hexlify(cert_info.tbs_certificate_bytes).decode())
# print('public_bytes:', cert_info.public_bytes(serialization.Encoding.PEM).decode())

print('fingerprint:', hexlify(cert_info.fingerprint(hashes.SHA1())).decode(), 'SHA1')
try:
    print('public_key:', cert_info.public_key())
except Exception as e:
    print('public_key:', e)
print('signature_hash_algorithm:', cert_info.signature_hash_algorithm)
sys.exit()

print('\n# SUBJECT '.ljust(80, '='))
show('subject', cert_info.subject, lambda x: x.value)
print('\n# ISSUER '.ljust(80, '='))
show('issuer', cert_info.issuer, lambda x: x.value)
print('\n# EXTENSIONS '.ljust(80, '='))
show('ext', cert_info.extensions, xvalue)
print('\n# PUBLIC KEY '.ljust(80, '='))
print(cert_info.public_key())
show('pub', cert_info.public_key, xvalue)

# vim:set ft=python ai et ts=4 sts=4 sw=4 cc=80:EOF #