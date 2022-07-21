#date: 2022-07-21T17:10:09Z
#url: https://api.github.com/gists/39f99fa46fb2f03ab005eafee77780f0
#owner: https://api.github.com/users/clach04

import io
import socket
import struct
import enum


def write_string(buf, string):
    buf.write(string.encode('utf-8'))
    buf.write(b'\x00')


class Message:
    msg_type = NotImplemented

    def _serialize_type(self, buf):
        buf.write(self.msg_type)

    def _serialize(self, buf):
        pass

    def serialize(self):
        buf = io.BytesIO()

        self._serialize_type(buf)
        type_size = buf.tell()

        buf.write(b'\x00' * 4)
        self._serialize(buf)
        full_size = buf.tell()

        buf.seek(type_size)
        buf.write(struct.pack('!I', full_size - type_size))

        return buf.getvalue()

    def __bytes__(self):
        rv = self.serialize()
        print('  >', self)
        # print('  >', rv)
        return rv

    @classmethod
    def deserialize(cls, buf):
        rv = cls()
        rv._deserialize(buf)
        return rv

    def _deserialize(self, buf):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}({self.values_repr()})'

    def values_repr(self):
        return ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())


class ResponseMessage(Message):
    pass


class StartupMessage(Message):
    def __init__(self, user, database=None):
        self.user = user
        self.database = database

    def _serialize_type(self, buf):
        pass

    def _serialize(self, buf):
        buf.write(struct.pack('!HH', 3, 0))
        write_string(buf, 'user')
        write_string(buf, self.user)
        if self.database is not None:
            write_string(buf, 'database')
            write_string(buf, self.database)
        buf.write(b'\x00')


class Authentication(ResponseMessage):
    msg_type = b'R'

    @classmethod
    def deserialize(self, buf):
        rv = authentication_by_type[struct.unpack('!I', buf[:4])[0]]()
        rv._deserialize(buf)
        return rv


class AuthenticationOK(Authentication):
    pass


class ParameterStatus(ResponseMessage):
    msg_type = b'S'

    def __init__(self):
        self.name = self.value = ''

    def _deserialize(self, buf):
        values = bytes(buf).split(b'\x00')
        self.name = values[0].decode('utf-8')
        self.value = values[1].decode('utf-8')


class BackendKeyData(ResponseMessage):
    msg_type = b'K'

    def _deserialize(self, buf):
        self.processID = struct.unpack('!i', buf[:4])[0]
        self.secretKey = struct.unpack('!i', buf[4:])[0]


class ReadyForQuery(ResponseMessage):
    msg_type = b'Z'

    class Type(enum.Enum):
        idle = b'I'
        in_trans = b'T'
        err_trans = b'E'

        def __repr__(self):
            return f'{self.__class__.__name__}.{self.name}'

    def _deserialize(self, buf):
        self.status = self.Type(buf[:1])


class Query(Message):
    msg_type = b'Q'

    def __init__(self, query):
        self.query = query

    def _serialize(self, buf):
        write_string(buf, self.query)


class RowDescription(ResponseMessage):
    msg_type = b'T'

    def __init__(self):
        self.fields = []

    def _deserialize(self, buf):
        offset = 2
        for i in range(struct.unpack('!H', buf[:2])[0]):
            field = {}
            pos = len(buf)
            for pos in range(offset, len(buf)):
                if buf[pos] == 0:
                    break
            field['name'] = bytes(buf[offset:pos]).decode('utf-8')
            field['table_oid'] = struct.unpack('!i', buf[pos + 1:pos + 5])[0]
            field['attribute_num'] = struct.unpack('!h', buf[pos + 5:pos + 7])[0]
            field['type_oid'] = struct.unpack('!i', buf[pos + 7:pos + 11])[0]
            field['type_size'] = struct.unpack('!h', buf[pos + 11:pos + 13])[0]
            field['type_modifier'] = struct.unpack('!i', buf[pos + 13:pos + 17])[0]
            field['format_code'] = struct.unpack('!h', buf[pos + 17:pos + 19])[0]
            self.fields.append(field)
            offset = pos + 19


class DataRow(ResponseMessage):
    msg_type = b'D'

    def __init__(self):
        self.values = []

    def _deserialize(self, buf):
        offset = 2
        for i in range(struct.unpack('!h', buf[:2])[0]):
            size = struct.unpack('!i', buf[offset:offset + 4])[0]
            if size < 0:
                self.values.append(None)
            else:
                self.values.append(bytes(buf[offset + 4:offset + 4 + size]))
            offset += 4 + size


class CommandComplete(ResponseMessage):
    msg_type = b'C'

    def _deserialize(self, buf):
        self.tag = bytes(buf).split(b'\x00')[0].decode('utf-8')


class ErrorResponse(ResponseMessage):
    msg_type = b'E'

    types = {
        b'S': 'severity',
        b'V': 'severity',
        b'C': 'code',
        b'M': 'message',
        b'D': 'detail',
        b'H': 'hint',
        b'P': 'position',
        b'p': 'internal_position',
        b'q': 'internal_query',
        b'W': 'where',
        b's': 'schema_name',
        b't': 'table_name',
        b'c': 'column_name',
        b'd': 'data_type_name',
        b'n': 'constraint_name',
        b'F': 'file',
        b'L': 'line',
        b'R': 'routine',
    }

    def _deserialize(self, buf):
        for field in bytes(buf).split(b'\x00')[:-2]:
            attr = self.types.get(field[:1])
            if attr is not None:
                setattr(self, attr, field[1:].decode('utf-8'))


class Parse(Message):
    msg_type = b'P'

    def __init__(self, query, statement_name=''):
        self.query = query
        self.statement_name = statement_name
        self.param_types = []

    def _serialize(self, buf):
        write_string(buf, self.statement_name)
        write_string(buf, self.query)
        buf.write(struct.pack('!h', len(self.param_types)))
        for t in self.param_types:
            buf.write(struct.pack('!i', t))


class ParseComplete(ResponseMessage):
    msg_type = b'1'


class Describe(Message):
    msg_type = b'D'
    target = NotImplemented

    def __init__(self, name=''):
        self.name = name

    def _serialize(self, buf):
        buf.write(self.target)
        write_string(buf, self.name)


class DescribeStatement(Describe):
    target = b'S'


class ParameterDescription(ResponseMessage):
    msg_type = b't'

    def __init__(self):
        self.parameters = []

    def _deserialize(self, buf):
        pass


class Bind(Message):
    msg_type = b'B'

    def __init__(self, portal='', statement='', parameters=None):
        self.portal = portal
        self.statement = statement
        if parameters is None:
            parameters = []
        self.parameters = parameters

    def _serialize(self, buf):
        write_string(buf, self.portal)
        write_string(buf, self.statement)
        buf.write(struct.pack('!h', 0))
        buf.write(struct.pack('!h', len(self.parameters)))
        for param in self.parameters:
            param = str(param).encode('utf-8')
            buf.write(struct.pack('!i', len(param)))
            buf.write(param)
        buf.write(struct.pack('!h', 0))


class BindComplete(ResponseMessage):
    msg_type = b'2'


class DescribePortal(Describe):
    target = b'P'


class Execute(Message):
    msg_type = b'E'

    def __init__(self, portal='', limit=0):
        self.portal = portal
        self.limit = limit

    def _serialize(self, buf):
        write_string(buf, self.portal)
        buf.write(struct.pack('!i', self.limit))


class PortalSuspended(ResponseMessage):
    msg_type = b's'


class Sync(Message):
    msg_type = b'S'


class Flush(Message):
    msg_type = b'H'


class NoticeResponse(ResponseMessage):
    msg_type = b'N'

    def _deserialize(self, buf):
        self.fields = {}
        for field in bytes(buf).split(b'\x00'):
            if field:
                field_type = struct.unpack('!b', field[:1])[0]
                value = field[1:].decode('utf-8')
                self.fields[field_type] = value


class Close(Message):
    msg_type = b'C'

    def __init__(self, name, close_type='S'):
        self.close_type = close_type
        self.name = name

    def _serialize(self, buf):
        buf.write(self.close_type.encode('utf-8'))
        write_string(buf, self.name)


class CloseComplete(ResponseMessage):
    msg_type = b'3'


class NoData(ResponseMessage):
    msg_type = b'n'


class CancelRequest(Message):
    def __init__(self, key_data):
        self.key_data = key_data

    def _serialize_type(self, buf):
        pass

    def _serialize(self, buf):
        buf.write(struct.pack(
            '!iii', 80877102,
            self.key_data.processID, self.key_data.secretKey))


class Terminate(Message):
    msg_type = b'X'


messages_by_type = dict(
    (cls.msg_type, cls)
    for cls in locals().values()
    if isinstance(cls, type) and issubclass(cls, ResponseMessage) and
    cls.msg_type is not None
)

authentication_by_type = {
    0: AuthenticationOK,
}


def deserialize(data):
    buf = memoryview(data)

    while buf:
        msg_type = bytes(buf[:1])
        msg_size = struct.unpack('!I', buf[1:5])[0]
        payload = buf[5:msg_size + 1]
        cls = messages_by_type.get(msg_type)
        if cls is None:
            print('<  ', 'skipping:', bytes(buf[:msg_size + 1]))
        else:
            rv = cls.deserialize(payload)
            print('<  ', rv)
            yield rv
        buf = buf[msg_size + 1:]


def read_until_ready(sock, num=1, expect=ReadyForQuery):
    rv = None
    while num > 0:
        data = sock.recv(65536)
        if not data:
            break
        for msg in deserialize(data):
            if isinstance(msg, expect):
                num -= 1
                rv = msg
    return rv


def startup(host, port, user, database):
    print('=' * 79)
    print('Startup and clear table')
    print()

    sock = socket.socket()
    sock.settimeout(4)
    sock.connect((host, port))
    sock.send(bytes(StartupMessage(user, database)))
    key_data = read_until_ready(sock, expect=BackendKeyData)

    sock.send(bytes(Query('DELETE FROM table1')))
    resp = read_until_ready(sock, expect=(ErrorResponse, CommandComplete))
    if isinstance(resp, ErrorResponse):
        sock.send(
            bytes(Query('CREATE TABLE table1 (id INTEGER PRIMARY KEY )')))
        read_until_ready(sock)

    return sock, key_data, (host, port)


def main(sock, key_data, endpoint):
    print()
    print('=' * 79)
    print('Simple query')
    print()

    sock.send(bytes(Query('INSERT INTO table1 VALUES (1)')))
    read_until_ready(sock)

    print()
    print('=' * 79)
    print('Pipelining')
    print()

    sock.send(
        bytes(Query('INSERT INTO table1 VALUES (2)')) +
        bytes(Query('INSERT INTO table1 VALUES (3)')) +
        bytes(Query('INSERT INTO table1 VALUES (4)'))
    )
    read_until_ready(sock, 3)

    print()
    print('=' * 79)
    print('Multiple commands')
    print()

    sock.send(bytes(Query(
        'INSERT INTO table1 VALUES (5);'
        'INSERT INTO table1 VALUES (6);'
        'INSERT INTO table1 VALUES (7);'
    )))
    read_until_ready(sock)

    print()
    print('=' * 79)
    print('Implicit Transaction Failed')
    print()

    sock.send(bytes(Query(
        'INSERT INTO table1 VALUES (8);'
        'INSERT INTO table1 VALUES (8);'
        'INSERT INTO table1 VALUES (9);'
    )))
    read_until_ready(sock)

    print()
    print('=' * 79)
    print('Explicit transaction')
    print()

    sock.send(bytes(Query('BEGIN')))
    read_until_ready(sock)
    sock.send(
        bytes(Query('INSERT INTO table1 VALUES (10)')) +
        bytes(Query('INSERT INTO table1 VALUES (10)')) +
        bytes(Query('INSERT INTO table1 VALUES (11)'))
    )
    read_until_ready(sock, 3)
    sock.send(bytes(Query('END')))
    read_until_ready(sock)

    print()
    print('=' * 79)
    print('Query Result')
    print()

    sock.send(bytes(Query('SELECT * FROM table1')))
    read_until_ready(sock)

    print()
    print('=' * 79)
    print('Extended Query')
    print()

    sock.send(
        bytes(Parse('SELECT * FROM table1', 'stmt1')) +
        # bytes(DescribeStatement('stmt1')) +
        bytes(Bind('portal1', 'stmt1')) +
        bytes(DescribePortal('portal1')) +
        bytes(Execute('portal1')) +
        bytes(Sync())
    )
    read_until_ready(sock)

    print()
    print('=' * 79)
    print('Implicit transaction failed with Extended Query')
    print()

    sock.send(
        bytes(Parse('SELECT 1/0', 'stmt2')) +
        bytes(Bind('portal2', 'stmt2')) +
        bytes(Execute('portal2')) +
        bytes(Sync())
    )
    read_until_ready(sock)

    print()
    print('=' * 79)
    print('ROLLBACK an implicit transaction')
    print()

    sock.send(
        bytes(Parse('SELECT now()', 'stmt3')) +
        bytes(Parse('ROLLBACK', 'stmt4')) +
        bytes(Bind('portal4', 'stmt4')) +
        bytes(Execute('portal4')) +
        bytes(Sync())
    )
    read_until_ready(sock)

    print()
    print('=' * 79)
    print('Parameter')
    print()

    sock.send(
        bytes(Parse('SELECT * FROM table1 WHERE id = $1', 'stmt5')) +
        bytes(Bind('portal5', 'stmt5', parameters=[4])) +
        bytes(Execute('portal5')) +
        bytes(Bind('portal6', 'stmt5', parameters=[2])) +
        bytes(Execute('portal6')) +
        bytes(Sync())
    )
    read_until_ready(sock)

    print()
    print('=' * 79)
    print('Server-side Cursor')
    print()

    sock.send(
        bytes(Parse('SELECT * FROM table1 ORDER BY id')) +
        bytes(Bind('cur1')) +
        bytes(Execute('cur1', limit=1)) +
        bytes(Execute('cur1', limit=1)) +
        bytes(Parse('MOVE 2 cur1')) +
        bytes(Bind()) +
        bytes(Execute()) +
        bytes(Execute('cur1')) +
        bytes(Sync())
    )
    read_until_ready(sock)

    print()
    print('=' * 79)
    print('Cancel')
    print()

    sock.send(
        bytes(Query('SELECT pg_sleep(3)'))
    )

    cancel = socket.socket()
    cancel.settimeout(0.5)
    cancel.connect(endpoint)
    cancel.send(bytes(CancelRequest(key_data)))
    read_until_ready(cancel)

    read_until_ready(sock)

    print()
    print('=' * 79)
    print('Bye')
    print()

    sock.send(bytes(Terminate()))
    read_until_ready(sock)
    sock.close()


if __name__ == '__main__':
    main(*startup('localhost', 5432, 'postgres', 'postgres'))
