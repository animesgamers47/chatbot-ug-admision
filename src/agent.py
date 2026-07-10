from src.preprocess import limpiar_texto
from src.classifier import Classifier
from src.entity_extractor import extraer_entidades

DEBUG = False


class Agent:
    def __init__(self, vectorizer):
        self.classifier = Classifier(vectorizer)

    def responder(self, consulta):
        if not consulta or not consulta.strip():
            return 'Por favor, escribe una consulta válida.'
        try:
            consulta_proc = limpiar_texto(consulta, filtro_guayaquil=True)
            if not consulta_proc:
                return 'No pude procesar tu consulta. ¿Puedes intentar con otras palabras?'
            tag, _ = self.classifier.detectar_intencion(consulta_proc)
            respuesta = self.classifier.obtener_respuesta(tag)
            entidades = extraer_entidades(consulta)
            if DEBUG and entidades:
                respuesta += f'\n\n[Entidades: {entidades}]'
            return respuesta
        except Exception:
            return 'Ocurrió un error procesando tu consulta.'


def iniciar_consola(vectorizer):
    agent = Agent(vectorizer)
    print('\n🤖 Chatbot UG - Admisión y Nivelación')
    print('Escribe "salir" para terminar.\n')
    while True:
        q = input('🧑 Tú: ').strip()
        if q.lower() in ('salir', 'exit', 'quit'):
            print('🤖 Chatbot: ¡Hasta luego!')
            break
        print(f'🤖 Chatbot: {agent.responder(q)}\n')
