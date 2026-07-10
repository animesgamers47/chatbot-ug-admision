import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')

ACENTOS = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n', 'ü': 'u'}


def normalizar_acentos(texto):
    for k, v in ACENTOS.items():
        texto = texto.replace(k, v)
    return texto


def limpiar_texto(texto, filtro_guayaquil=True):
    texto = texto.lower()
    texto = normalizar_acentos(texto)
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    tokens = texto.split()
    resultado = []
    for t in tokens:
        if t in stop_words:
            continue
        if len(t) <= 1:
            continue
        if filtro_guayaquil and t == 'guayaquil':
            continue
        resultado.append(stemmer.stem(t))
    return ' '.join(resultado)
