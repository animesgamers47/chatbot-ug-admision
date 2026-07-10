import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from src.vectorizer import Vectorizer
from src.agent import Agent

vec = Vectorizer()
agent = Agent(vec)


def chatbot_fn(message, history):
    return agent.responder(message)


with gr.Blocks(title='Chatbot UG - Admision y Nivelacion') as demo:
    gr.Markdown('''
    # Chatbot de Admision y Nivelacion - UG
    **Universidad de Guayaquil** | Agente conversacional basado en PLN clasico

    Pregunta sobre requisitos, fechas, carreras, costos y mas.
    ''')
    gr.ChatInterface(
        fn=chatbot_fn,
        title='',
    )

if __name__ == '__main__':
    demo.launch()
