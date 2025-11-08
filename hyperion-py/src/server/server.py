from flask import Flask, request

class Server:
    """
    Spins up a flask server that accepts /train and /predict
    """
    def __init__(self):
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/train', methods=['POST'])
        def train():
            """
            Endpoint for training models.
            """
            if request.method == 'POST':
                ticker, interval, period = request.json['ticker'], request.json['interval'], request.json['period']



    def run(self):
        self.app.run(host='0.0.0.0', port=5000, debug=True)