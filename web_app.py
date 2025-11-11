import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, this is a placeholder page for the THG Discord bot."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environget('PORT', 5000)))
