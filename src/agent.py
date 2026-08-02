import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import limpiar_texto
from src.classifier import Classifier, UMBRAL_CONFIANZA, MARGEN_MINIMO
from src.entity_extractor import extraer_entidades

DEBUG = False


def enriquecer_respuesta(respuesta, entidades):
    if not entidades:
        return respuesta
    fragmentos = []
    if 'carreras' in entidades:
        carreras = ', '.join(entidades['carreras'][:2])
        fragmentos.append(f'Veo que preguntas sobre {carreras}. '
                          f'Consulta la oferta completa en admision.ug.edu.ec/oferta-ug/')
    if 'montos' in entidades:
        fragmentos.append(f'Mencionaste el valor {entidades["montos"][0]}; '
                          f'verifica siempre los valores oficiales vigentes.')
    if 'fechas' in entidades:
        fragmentos.append(f'Recuerda la fecha mencionada: {entidades["fechas"][0]}.')
    if 'cédulas' in entidades:
        fragmentos.append('No almaceno tu cédula; úsala solo en los trámites oficiales de la UG.')
    return respuesta + '\n\n' + ' '.join(fragmentos)


class Agent:
    def __init__(self, vectorizer):
        self.classifier = Classifier(vectorizer)

    def _procesar(self, consulta):
        if not consulta or not consulta.strip():
            return None
        consulta_proc = limpiar_texto(consulta, filtro_guayaquil=True)
        return consulta_proc if consulta_proc else None

    def responder(self, consulta):
        try:
            consulta_proc = self._procesar(consulta)
            if consulta_proc is None:
                return 'Por favor, escribe una consulta válida.'
            tag, _ = self.classifier.detectar_intencion(consulta_proc)
            respuesta = self.classifier.obtener_respuesta(tag)
            entidades = extraer_entidades(consulta)
            respuesta = enriquecer_respuesta(respuesta, entidades)
            if DEBUG and entidades:
                respuesta += f'\n\n[Entidades: {entidades}]'
            return respuesta
        except Exception:
            return 'Ocurrió un error procesando tu consulta.'

    def responder_con_detalle(self, consulta):
        try:
            consulta_proc = self._procesar(consulta)
            if consulta_proc is None:
                return 'Por favor, escribe una consulta válida.', None, None
            tag, score, segundo, score2 = self.classifier.diagnosticar(consulta_proc)
            tag = tag if score >= UMBRAL_CONFIANZA and \
                (score - score2) >= MARGEN_MINIMO else 'fallback'
            respuesta = self.classifier.obtener_respuesta(tag)
            entidades = extraer_entidades(consulta)
            respuesta = enriquecer_respuesta(respuesta, entidades)
            if DEBUG and entidades:
                respuesta += f'\n\n[Entidades: {entidades}]'
            return respuesta, tag, score
        except Exception:
            return 'Ocurrió un error procesando tu consulta.', None, None


def iniciar_consola(vectorizer=None):
    if vectorizer is None:
        from src.vectorizer import Vectorizer
        vectorizer = Vectorizer()
    agent = Agent(vectorizer)
    print('\n🤖 Chatbot UG - Admisión y Nivelación')
    print('Escribe "salir" para terminar.\n')
    while True:
        q = input('🧑 Tú: ').strip()
        if q.lower() in ('salir', 'exit', 'quit'):
            print('🤖 Chatbot: ¡Hasta luego!')
            break
        print(f'🤖 Chatbot: {agent.responder(q)}\n')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    iniciar_consola()
