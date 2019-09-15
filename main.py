from http.server import BaseHTTPRequestHandler, HTTPServer
from call_clustering import infer
import time

hostName = "0.0.0.0"
hostPort = 80

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(bytes("<p>You accessed path: %s</p>" % self.path, "utf-8"))

    def do_POST(self):
        print("incomming POST: ", self.path)
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        result = infer(post_data)
        self.send_header("content-type", "application/json")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(bytes(result, "utf-8"))

myServer = HTTPServer((hostName, hostPort), MyServer)
print(time.asctime(), "Server Starts - %s:%s" % (hostName, hostPort))

myServer.serve_forever()
