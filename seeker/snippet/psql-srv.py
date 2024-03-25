#date: 2024-03-25T16:59:46Z
#url: https://api.github.com/gists/b2449a565c0e95e94627a7f28bcbe531
#owner: https://api.github.com/users/ddurham2

# th30z@u1310:[Desktop]$ psql -h localhost -p 55432
# "**********"Password: "**********"
# psql (9.1.10, server 0.0.0)
# WARNING: psql version 9.1, server version 0.0.
#          Some psql features might not work.
# Type "help" for help.
# 
# th30z=> select foo;
#  a | b 
# ---+---
#  1 | 2
#  3 | 4
#  5 | 6
# (3 rows)

from io import BytesIO
import socketserver
import struct

PSQL_FE_MESSAGES = {
  b'p': "**********"
  b'Q': "Simple query",
  b'P': "Parse",
  b'B': "Bind",
  b'E': "Execute",
  b'D': "Describe",
  b'C': "Close",
  b'H': "Flush",
  b'S': "Sync",
  b'F': "Function call",
  b'd': "Copy data",
  b'c': "Copy completion",
  b'f': "Copy failure",
  b'X': "Termination",
}

class IntField:
  def __init__(self, name):
    self.name = name
    self.type_id = 23
    self.type_size = 4

class PgBuffer(object):
  def __init__(self, stream):
    self.buffer = stream

  def read_byte(self):
    return self.read_bytes(1)

  def read_bytes(self, n):
    data = self.buffer.read(n)
    if not data:
      raise Exception("No data")
    return data

  def read_int32(self):
    data = self.read_bytes(4)
    return struct.unpack("!i", data)[0]

  def read_parameters(self, n):
    data = self.read_bytes(n)
    return data.split(b'\x00')

  def write_byte(self, value):
    self.buffer.write(value)

  def write_bytes(self, value):
    self.buffer.write(value)

  def write_int16(self, value):
    self.buffer.write(struct.pack("!h", value))

  def write_int32(self, value):
    self.buffer.write(struct.pack("!i", value))

  def write_string(self, value):
    self.buffer.write(value.encode() if isinstance(value, str) else value)
    self.buffer.write(b'\x00')

  def write_parameters(self, kvs):
    data = ''.join(['%s\x00%s\x00' % kv in kvs])
    self.buffer.write_int32(4 + len(data))
    self.buffer.write(data)

class PsqlHandler(socketserver.StreamRequestHandler):
  def handle(self):
    self._pgbuf = PgBuffer(self.rfile)
    self.wbuf = PgBuffer(self.wfile)

    try:
      # Handshake
      self.read_ssl_request()
      self.send_notice()
      self.read_startup_message()

      # Auth
      self.send_authentication_request()
      self.read_authentication()
      self.send_authentication_ok()

      while True:
        self.send_ready_for_query()

        # Read Message
        type_code = self._pgbuf.read_byte()
        print(PSQL_FE_MESSAGES.get(type_code))

        if type_code == b'Q':
          msglen = self._pgbuf.read_int32()
          sql = self._pgbuf.read_bytes(msglen - 4)
          self.query(sql)
        elif type_code == b'X':
          break
    except Exception as e:
      print(e)
      raise e

  def read_ssl_request(self):
    msglen = self._pgbuf.read_int32()
    sslcode = self._pgbuf.read_int32()
    if msglen != 8 and sslcode != 80877103:
      raise Exception("Unsupported SSL request")

  def read_startup_message(self):
    msglen = self._pgbuf.read_int32()
    version = self._pgbuf.read_int32()
    v_maj = version >> 16
    v_min = version & 0xffff
    msg = self._pgbuf.read_parameters(msglen - 8)
    print('PSQ %d.%d - %s' % (v_maj, v_min, msg))

  def read_authentication(self):
    type_code = self._pgbuf.read_byte()
    if type_code != b"p":
      self.send_error("FATAL", "28000", "authentication failure")
      raise Exception("Only 'Password' auth is supported, got %r" % type_code)

    msglen = self._pgbuf.read_int32()
    password = "**********"
    print(password)

  def send_notice(self):
    self.wfile.write(b'N')

  def send_authentication_request(self):
    self.wfile.write(struct.pack("!cii", b'R', 8, 3))

  def send_authentication_ok(self):
    self.wfile.write(struct.pack("!cii", b'R', 8, 0))

  def send_ready_for_query(self):
    self.wfile.write(struct.pack("!cic", b'Z', 5, b'I'))

  def send_command_complete(self, tag):
    self.wfile.write(struct.pack("!ci", b'C', 4 + len(tag)))
    self.wfile.write(tag)

  def send_error(self, severity, code, message):
    buf = PgBuffer(BytesIO())
    buf.write_byte(b'S')
    buf.write_string(severity)
    buf.write_byte(b'C')
    buf.write_string(code)
    buf.write_byte(b'M')
    buf.write_string(message)
    buf = buf.buffer.getvalue()

    self.wbuf.write_byte(b'E')
    self.wbuf.write_int32(4 + len(buf))
    self.wbuf.write_string(buf)

  def send_row_description(self, fields):
    buf = PgBuffer(BytesIO())
    for field in fields:
      buf.write_string(field.name)
      buf.write_int32(0)    # Table ID
      buf.write_int16(0)    # Column ID
      buf.write_int32(field.type_id)
      buf.write_int16(field.type_size)
      buf.write_int32(-1)   # type modifier
      buf.write_int16(0)    # text format code
    buf = buf.buffer.getvalue()

    self.wbuf.write_byte(b'T')
    self.wbuf.write_int32(6 + len(buf))
    self.wbuf.write_int16(len(fields))
    self.wbuf.write_bytes(buf)

  def send_row_data(self, rows):
    for row in rows:
      buf = PgBuffer(BytesIO())
      for field in row:
        v = b'%r' % field
        buf.write_int32(len(v))
        buf.write_bytes(v)
      buf = buf.buffer.getvalue()

      self.wbuf.write_byte(b'D')
      self.wbuf.write_int32(6 + len(buf))
      self.wbuf.write_int16(len(row))
      self.wbuf.write_bytes(buf)

  def query(self, sql):
    fields = [IntField('a'), IntField('b')]
    rows = [[1, 2], [3, 4], [5, 6]]

    self.send_row_description(fields)
    self.send_row_data(rows)

    self.send_command_complete(b'SELECT\x00')

class TcpServer(socketserver.TCPServer):
  allow_reuse_address = True

if __name__ == '__main__':
  server = TcpServer(("localhost", 55432), PsqlHandler)
  print('server running, try: $ psql -h localhost -p 55432')
  try:
    server.serve_forever()
  except:
    server.shutdown()own()