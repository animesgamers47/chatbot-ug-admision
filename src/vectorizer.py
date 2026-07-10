import json
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import limpiar_texto


class Vectorizer:
    def __init__(self, intents_path='data/intents.json'):
        with open(intents_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.utterances = []
        self.tags = []
        self.response_map = {}

        for intent in data['intents']:
            self.response_map[intent['tag']] = intent['responses']
            for pattern in intent['patterns']:
                self.utterances.append(pattern)
                self.tags.append(intent['tag'])

        utterances_proc = [limpiar_texto(u, False) for u in self.utterances]

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.X = self.vectorizer.fit_transform(utterances_proc)
