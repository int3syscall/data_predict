from http.server import HTTPServer, BaseHTTPRequestHandler
from test1_working_model_copy import predict
from fetch_data import fetch_datas


HOST = "127.0.0.1"
PORT = 8055

def get_prediction():

    last_day_data = fetch_datas("localhost", 3306, "laveraluser", "testdata2s")
    prediction = predict(last_day_data)
    prediction = str(prediction).replace("},{", "},\n{")

    return prediction



class simpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/data":

            response = get_prediction()

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Language', 'en')
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))
            return

address = (HOST,PORT)
httpd = HTTPServer(address, simpleHTTPRequestHandler)
httpd.serve_forever(0.1)