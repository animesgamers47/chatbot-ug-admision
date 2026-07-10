import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


UMBRAL_CONFIANZA = 0.35


class Classifier:
    def __init__(self, vectorizer):
        self.vec = vectorizer

    def detectar_intencion(self, consulta_proc):
        consulta_vec = self.vec.vectorizer.transform([consulta_proc])
        similitudes = cosine_similarity(consulta_vec, self.vec.X).flatten()
        idx_max = np.argmax(similitudes)
        max_sim = similitudes[idx_max]
        if max_sim >= UMBRAL_CONFIANZA:
            return self.vec.tags[idx_max], max_sim
        return 'fallback', max_sim

    def obtener_respuesta(self, tag):
        if tag == 'fallback':
            return ('Lo siento, no entendí tu consulta. '
                    '¿Puedes reformularla? Puedo ayudarte con temas de '
                    'admisión, nivelación, requisitos, costos y más sobre la UG.')
        respuestas = self.vec.response_map.get(tag, ['No tengo respuesta.'])
        return np.random.choice(respuestas)
