from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import soundfile as sf
import io

app = Flask(__name__)

# Your existing TTS code and model loading here...

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)