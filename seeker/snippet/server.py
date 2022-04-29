#date: 2022-04-29T17:13:34Z
#url: https://api.github.com/gists/f995396b312cf3aac92e14d85e4eb9ae
#owner: https://api.github.com/users/tcyrus

from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import json
import semantle_solver as ss

page = open('index.html','rb').read()

class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        #logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        #self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))
        self.wfile.write(page)

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        try:
            jsondata = json.loads(post_data)
            words = ss.solve_semantle([[j[0],float(j[1])] for j in jsondata if j[0] != ""])
            if len(words) > 100:
                response = '> 100 words'
            else:
                response = str(words)
        except Exception as e:
            response = 'error: %s'%str(e)
        #logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
        #        str(self.path), str(self.headers), post_data.decode('utf-8'))
        self._set_response()
        self.wfile.write(response.encode())
        #self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=S, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
