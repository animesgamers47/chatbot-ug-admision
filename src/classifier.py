import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


UMBRAL_CONFIANZA = 0.40
MARGEN_MINIMO = 0.05


class Classifier:
    def __init__(self, vectorizer):
        self.vec = vectorizer
        self.tag_idx = {}
        for idx, tag in enumerate(self.vec.tags):
            self.tag_idx.setdefault(tag, []).append(idx)
        self.tags_unicos = list(self.tag_idx.keys())

    def _scores_por_intent(self, consulta_proc):
        consulta_vec = self.vec.vectorizer.transform([consulta_proc])
        similitudes = cosine_similarity(consulta_vec, self.vec.X).flatten()
        scores = {}
        for tag, idxs in self.tag_idx.items():
            scores[tag] = float(similitudes[idxs].max())
        return scores

    def detectar_intencion(self, consulta_proc):
        scores = self._scores_por_intent(consulta_proc)
        ranking = sorted(scores.items(), key=lambda kv: -kv[1])
        mejor, s_mejor = ranking[0]
        segundo, s_segundo = ranking[1]
        if s_mejor >= UMBRAL_CONFIANZA and (s_mejor - s_segundo) >= MARGEN_MINIMO:
            return mejor, s_mejor
        return 'fallback', s_mejor

    def diagnosticar(self, consulta_proc):
        scores = self._scores_por_intent(consulta_proc)
        ranking = sorted(scores.items(), key=lambda kv: -kv[1])
        mejor, s_mejor = ranking[0]
        segundo, s_segundo = ranking[1]
        return mejor, s_mejor, segundo, s_segundo

    def obtener_respuesta(self, tag):
        respuestas = self.vec.response_map.get(tag)
        if not respuestas:
            return ('Lo siento, no entendí tu consulta. '
                    '¿Puedes reformularla? Puedo ayudarte con temas de '
                    'admisión, nivelación, requisitos, costos y más sobre la UG.')
        return np.random.choice(respuestas)
