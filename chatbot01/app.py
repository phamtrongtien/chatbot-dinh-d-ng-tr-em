from flask import Flask, render_template, request, jsonify
from main import ImprovedChatBot, NltkUtils, Training, Model
import json

app = Flask(__name__)

# Load intents
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Create chatbot instance and train
chatbot = ImprovedChatBot(intents)
chatbot.train(lambda_reg=0.012, epochs=10000, learning_rate=0.008)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        intent_tag = chatbot.predict(message)
        response = chatbot.get_response(intent_tag)
        return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
