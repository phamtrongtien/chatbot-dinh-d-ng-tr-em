# chatbot.py
import json
import random
from model import Model
from train import Training
from nltk import NltkUtils
import numpy as np

class ImprovedChatBot:
    def __init__(self, intents):
        self.intents = intents
        self.model = None
        self.tags = None
        self.all_words = None

    def train(self, lambda_reg=0.012, epochs=2000, learning_rate=0.008):
        all_words = []
        tags = []
        documents = []

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                pattern_words = NltkUtils.clean_up_sentence(pattern)
                all_words.extend(pattern_words)
                documents.append((pattern_words, intent['tag']))
                if intent['tag'] not in tags:
                    tags.append(intent['tag'])

        all_words = sorted(set(all_words))
        tags = sorted(set(tags))

        X_train = []
        y_train = []

        for document in documents:
            bag = NltkUtils.bag_of_words(' '.join(document[0]), all_words)
            X_train.append(bag)
            label = tags.index(document[1])
            y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        num_classes = len(tags)
        num_features = X_train.shape[1]

        self.model = Model(num_features, num_classes)

        for epoch in range(epochs):
            loss_softmax = Training.train_softmax(X_train, y_train, self.model, learning_rate, lambda_reg)

            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Loss (Softmax): {loss_softmax}')

        self.tags = tags
        self.all_words = all_words

    def predict(self, sentence):#phương thức dự đoán trong một mô hình học máy.
        bag = NltkUtils.bag_of_words(sentence, self.all_words)

        scores_softmax = np.dot(bag, self.model.weights_softmax) + self.model.biases_softmax
        probabilities_softmax = Training.softmax(scores_softmax)
        predicted_class = np.argmax(probabilities_softmax)

        scores_logistic = np.dot(bag, self.model.weights_logistic) + self.model.bias_logistic
        probability_logistic = Training.sigmoid(scores_logistic)

        # print("Predicted Class:", self.tags[predicted_class])
        # print("Logistic Probability:", probability_logistic)

        if probability_logistic > 0.5:
            return self.tags[predicted_class]
        else:
            return "Uncertain"

    def get_response(self, intent_tag):
        # print("Intent Tag:", intent_tag)
        for intent in self.intents['intents']:
            if intent['tag'] == intent_tag:
                if intent_tag == 'ask_stages':
                    return "Theo kiến thức của tôi, thường trẻ sơ sinh có 5 giai đoạn từ lúc mới sinh đến 12 tháng."
                else:
                    return random.choice(intent['responses'])
        return "Xin lỗi, tôi không hiểu câu hỏi của bạn.Bạn có thể hỏi lại được không?"

    def chat(self):
        print("Go! Bot is running")

        while True:
            message = input("Bạn:")
            intent_tag = self.predict(message)
            response = self.get_response(intent_tag)
            print("Bot:" + response)

if __name__ == "__main__":
    with open('intents.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
    chatbot = ImprovedChatBot(intents)
    chatbot.train(lambda_reg=0.012, epochs=10000, learning_rate=0.008)
    chatbot.chat()


