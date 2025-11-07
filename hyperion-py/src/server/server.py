from flask import Flask, request

class FlaskServer:
    """

    """
    def __init__(self):
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/train', methods=['GET', 'POST'])
        def train():
            """
            Endpoint for training models.
            """
            if request.method == 'POST':
                ticker = request.args['ticker']
                print(ticker)

    def run(self):
        self.app.run(host='0.0.0.0', port=5000, debug=True)