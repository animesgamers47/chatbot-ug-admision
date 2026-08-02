import sys
import os
import csv
import time
import base64
import random
from functools import partial
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from src.vectorizer import Vectorizer
from src.agent import Agent
from src.preprocess import limpiar_texto

RAIZ = Path(__file__).resolve().parent.parent
ASSETS = RAIZ / 'assets'
DATA = RAIZ / 'data'
LOGO_PATH = ASSETS / 'LogoUGcolor.png'
FEEDBACK_PATH = DATA / 'feedback.csv'

vec = Vectorizer()
agent = Agent(vec)

RESPUESTA_INVALIDA = 'Por favor, escribe una consulta válida.'


def data_uri(path):
    mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
    return f'data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}'


LOGO_URI = data_uri(LOGO_PATH) if LOGO_PATH.exists() else None
USER_AVATAR_PATH = str(ASSETS / 'user.svg')
BOT_AVATAR_PATH = str(LOGO_PATH)

CHIPS_INICIALES = [
    '¿Qué necesito para inscribirme?',
    '¿Cuándo es la evaluación de admisión?',
    '¿Cuánto cuesta la nivelación?',
    '¿Cómo asignan los cupos?',
]

POOL_PREGUNTAS = [
    '¿Qué necesito para inscribirme?',
    '¿Cuándo es la evaluación de admisión?',
    '¿Cuánto cuesta la nivelación?',
    '¿Cómo asignan los cupos?',
    '¿Qué carreras ofrece la UG?',
    '¿Cómo entro a Moodle?',
    '¿Dónde está el SIUG?',
    '¿Cuándo inicia la nivelación?',
    '¿Qué es la gratuidad?',
    '¿Cómo pido un reembolso?',
    '¿Cómo apruebo la nivelación?',
    '¿Qué nota necesito?',
    '¿Qué plataformas usa la UG?',
    '¿Cómo veo mi nota?',
    '¿Puedo estudiar siendo extranjero?',
    '¿Cómo obtengo una beca?',
    '¿Qué pasa si repruebo?',
    '¿Dónde queda la UG?',
    '¿Cuál es el número de la UG?',
    '¿Cómo me matriculo?',
]

FOLLOW_UPS = {
    'fallback': CHIPS_INICIALES,
    'saludo': ['¿Qué necesito para inscribirme?', '¿Cuándo es la evaluación?',
               '¿Qué carreras ofrece la UG?', '¿Cuánto cuesta la nivelación?'],
    'costos_aranceles': ['¿Cuánto cuesta la nivelación?',
                         '¿Cómo pido un reembolso?',
                         '¿Hay descuentos por discapacidad?',
                         '¿Qué es la gratuidad?'],
    'matricula_nivelacion': ['¿Cuándo inicia la nivelación?',
                             '¿Qué requisitos piden?',
                             '¿Cuánto cuesta la nivelación?',
                             '¿Qué plataformas usa la UG?'],
    'inscripcion_postulacion': ['¿Qué necesito para inscribirme?',
                                '¿Cuándo es el registro nacional?',
                                '¿Cómo asignan los cupos?',
                                '¿Qué carreras ofrece la UG?'],
    'registro_nacional': ['¿Qué es el registro nacional?',
                          '¿Qué necesito para postular?',
                          '¿Cuándo es la inscripción?',
                          '¿Cómo asignan los cupos?'],
    'cronograma_evaluaciones': ['¿De qué trata la prueba?',
                                '¿Qué necesito para la prueba?',
                                '¿Cómo asignan los cupos?',
                                '¿Cuándo es la nivelación?'],
    'evaluacion_admision': ['¿Cuándo es el examen de ingreso?',
                            '¿De qué trata la prueba?',
                            '¿Qué nota necesito?',
                            '¿Qué carreras ofrece la UG?'],
    'asignacion_cupos': ['¿Cómo acepto mi cupo?', '¿Qué nota necesito?',
                         '¿Qué es el puntaje de corte?',
                         '¿Qué pasa si pierdo la gratuidad?'],
    'aceptacion_cupo': ['¿Cómo asignan los cupos?', '¿Perdí mi cupo, qué hago?',
                        '¿Qué carreras ofrece la UG?',
                        '¿Qué necesito para nivelación?'],
    'requisitos_nivelacion': ['¿Cuánto cuesta la nivelación?',
                              '¿Cuándo inicia la nivelación?',
                              '¿Cómo me matriculo?',
                              '¿Qué plataformas usa la UG?'],
    'evaluacion_nivelacion': ['¿Cómo apruebo la nivelación?',
                              '¿Qué nota necesito?',
                              '¿Cómo veo mi nota?',
                              '¿Qué pasa si repruebo?'],
    'plataformas_virtuales': ['¿Cómo entro a Moodle?', '¿Dónde está el SIUG?',
                              '¿Cómo veo mi nota?',
                              '¿Cuánto cuesta la nivelación?'],
    'oferta_academica': ['¿Qué carreras de salud hay?',
                         '¿Cómo asignan los cupos?',
                         '¿Qué necesito para inscribirme?',
                         '¿Cuánto cuesta estudiar?'],
    'gratuidad_ug': ['¿Cómo pierdo la gratuidad?', '¿Quiénes pueden acceder?',
                     '¿Qué pasa si repruebo?', '¿Cuánto cuesta la matrícula?'],
    'perdida_gratuidad': ['¿Cómo recupero la gratuidad?',
                          '¿Qué pasa si repruebo?',
                          '¿Cuánto cuesta la segunda matrícula?',
                          '¿Cómo asignan los cupos?'],
    'descuentos_vulnerables': ['¿Cómo obtengo una beca?',
                               '¿Cuánto paga un estudiante con discapacidad?',
                               '¿Qué es la gratuidad?',
                               '¿Cuánto cuesta la matrícula?'],
    'reembolso_aranceles': ['¿Cómo pido un reembolso?',
                            '¿Puedo retirarme de la nivelación?',
                            '¿Cuánto cuesta la matrícula?',
                            '¿Qué es la gratuidad?'],
    'tipos_matricula': ['¿Qué pasa si me matriculo tarde?',
                        '¿Cuánto cuesta la matrícula?',
                        '¿Qué pasa si no pago?',
                        '¿Cómo asignan los cupos?'],
    'transferencia_estudiantes': ['¿Qué necesito para transferirme?',
                                  '¿Qué carreras ofrece la UG?',
                                  '¿Cuánto cuesta la matrícula?',
                                  '¿Qué requisitos piden?'],
    'requisitos_extranjeros': ['¿Puedo estudiar siendo extranjero?',
                               '¿Qué carreras ofrece la UG?',
                               '¿Cuánto cuesta la matrícula?',
                               '¿Qué necesito para inscribirme?'],
    'politicas_cupos': ['¿Cómo asignan los cupos?',
                        '¿Qué es el puntaje de corte?',
                        '¿Cómo acepto mi cupo?',
                        '¿Qué pasa si pierdo la gratuidad?'],
    'contacto': ['¿Cuál es el número de la UG?', '¿A qué correo escribo?',
                 '¿Dónde queda la UG?', '¿Qué carreras ofrece la UG?'],
    'despedida': ['¿Qué necesito para inscribirme?',
                  '¿Cuándo es la evaluación?', '¿Qué carreras ofrece la UG?',
                  '¿Cuánto cuesta la nivelación?'],
}

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Inter'), 'system-ui', 'sans-serif'],
).set(
    body_background_fill='#F4F6F9',
    background_fill_primary='#FFFFFF',
    background_fill_secondary='#EEF1F5',
    border_color_primary='#DCE1E8',
    border_color_accent='#1E3A5F',
    color_accent='#1E3A5F',
    color_accent_soft='#E3EAF2',
    body_text_color='#1C2733',
    body_text_color_subdued='#5A6B7C',
    link_text_color='#1E3A5F',
)

CSS = """
.ug-hero {
  position: relative;
  border-radius: 14px;
  overflow: hidden;
  padding: 20px 24px;
  margin: 4px 0 12px 0;
  background: linear-gradient(135deg, #142B47 0%, #1E3A5F 100%);
}
.ug-hero-inner { display: flex; align-items: center; gap: 18px; }
.ug-hero img.logo {
  height: 60px;
  width: auto;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
}
.ug-hero h1 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  letter-spacing: -0.3px;
  color: #FFFFFF;
}
.ug-hero p { margin: 2px 0; font-size: 13px; color: #B7C6D8; }
.ug-hero b { color: #FFFFFF; font-weight: 600; }
.ug-links a {
  margin-right: 16px;
  font-size: 12.5px;
  color: #9DB8D6 !important;
  text-decoration: none;
}
.ug-links a:hover { color: #FFFFFF !important; text-decoration: underline; }

#estado {
  font-size: 12.5px;
  color: #5A6B7C;
  margin: 2px 0 8px 2px;
}
#estado b { color: #1E3A5F; }

#chips-col { gap: 8px; min-width: 210px !important; }
.chips-title {
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 0.4px;
  text-transform: uppercase;
  color: #5A6B7C;
  margin-bottom: 4px;
}
.chip {
  border-radius: 9px !important;
  background: #142B47 !important;
  color: #FFFFFF !important;
  font-size: 12.5px !important;
  padding: 6px 12px !important;
  border: 1px solid #142B47 !important;
  box-shadow: none !important;
  justify-content: flex-start !important;
  text-align: left !important;
  white-space: normal !important;
  line-height: 1.35 !important;
}
.chip:hover {
  background: #1E3A5F !important;
  border-color: #1E3A5F !important;
}

#btn-col { gap: 8px; }
#enviar, #limpiar {
  font-size: 13px !important;
  padding: 7px 14px !important;
  min-height: 36px !important;
}

/* Burbujas de chat */
[data-testid="user"] .message {
  background: #142B47 !important;
  color: #FFFFFF !important;
  border-radius: 18px 18px 4px 18px !important;
}
[data-testid="assistant"] .message {
  background: var(--background-fill-secondary) !important;
  color: var(--body-text-color) !important;
  border-radius: 18px 18px 18px 4px !important;
  border: 1px solid var(--border-color-primary) !important;
}

.chatbot { border: 1px solid var(--border-color-primary) !important; border-radius: 14px !important; }

#pie {
  text-align: center;
  font-size: 12px;
  color: #5A6B7C;
  margin: 14px 0 4px 0;
}
#pie a { color: #1E3A5F; }

@media (max-width: 640px) {
  .ug-hero-inner { flex-direction: column; text-align: center; gap: 8px; }
  .ug-hero img.logo { height: 52px; }
  .ug-hero h1 { font-size: 17px; }
  .ug-links a { margin: 0 8px; }
  #chips-col { flex-direction: row; flex-wrap: wrap; gap: 6px; }
  #chips-col button { flex: 1 1 45%; }
  .chips-title { width: 100%; }
}
"""


def formato_detalle(tag, score):
    return ''


def responder(message):
    respuesta, tag, score = agent.responder_con_detalle(message)
    detalle = formato_detalle(tag, score)
    return respuesta, detalle


def handle_message(mensaje, history, browser_state):
    history = history if isinstance(history, list) else []
    if not mensaje or not mensaje.strip():
        history.append({'role': 'assistant', 'content': RESPUESTA_INVALIDA})
        return (history, 'Escribe una consulta válida.', '',
                *[gr.update(value=c) for c in CHIPS_INICIALES],
                CHIPS_INICIALES, history)
    respuesta, tag, score = agent.responder_con_detalle(mensaje)
    detalle = formato_detalle(tag, score)
    history.append({'role': 'user', 'content': mensaje})
    history.append({'role': 'assistant', 'content': respuesta})
    chips = FOLLOW_UPS.get(tag, CHIPS_INICIALES)
    return (history, detalle, '',
            *[gr.update(value=c) for c in chips], chips, history)


def enviar_chip(i, chips_state, history, browser_state):
    chips = chips_state if (isinstance(chips_state, list) and len(chips_state) == 4) else CHIPS_INICIALES
    return handle_message(chips[i], history, browser_state)


def registrar_feedback(*args):
    try:
        like_data = next((a for a in args if isinstance(a, gr.LikeData)), None)
        history = next((a for a in args if isinstance(a, list)), None)
        if like_data is None:
            return None
        idx = like_data.index
        valor = str(like_data.value)
        tag = ''
        if isinstance(idx, int) and isinstance(history, list) and 0 <= idx < len(history):
            mensaje_usuario = ''
            if idx >= 1 and isinstance(history[idx - 1], dict) and history[idx - 1].get('role') == 'user':
                mensaje_usuario = history[idx - 1].get('content') or ''
            if mensaje_usuario:
                proc = limpiar_texto(mensaje_usuario, filtro_guayaquil=True)
                if proc:
                    tag, _, _, _ = agent.classifier.diagnosticar(proc)
        DATA.mkdir(exist_ok=True)
        escribir_encabezado = not FEEDBACK_PATH.exists()
        with open(FEEDBACK_PATH, 'a', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            if escribir_encabezado:
                w.writerow(['timestamp', 'index', 'valor', 'intencion'])
            w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), idx, valor, tag])
    except Exception:
        pass
    return None


def limpiar_conversacion():
    return ([], 'Escribe tu consulta para comenzar.',
            *[gr.update(value=c) for c in CHIPS_INICIALES],
            CHIPS_INICIALES, [])


def restaurar(browser_state):
    return browser_state if isinstance(browser_state, list) else []


def sugerencias_aleatorias():
    elegidas = random.sample(POOL_PREGUNTAS, 4)
    return (*[gr.update(value=q) for q in elegidas], elegidas)


logo_html = f'<img class="logo" src="{LOGO_URI}" alt="Logo UG"/>' if LOGO_URI else ''

header_html = f'''
<div class="ug-hero">
  <div class="ug-hero-inner">
    {logo_html}
    <div>
      <h1>Chatbot de Admisión y Nivelación - UG</h1>
      <p><b>Universidad de Guayaquil</b> · Agente conversacional</p>
      <p class="ug-links">
        <a href="https://admision.ug.edu.ec/admision/" target="_blank">Portal oficial</a>
        <a href="https://admision.ug.edu.ec/oferta-ug/" target="_blank">Oferta académica</a>
        <a href="https://admision.ug.edu.ec/nivelacion/" target="_blank">Nivelación</a>
      </p>
    </div>
  </div>
</div>
'''

footer_html = '''
<div id="pie">
  Chatbot informativo sobre el proceso de admisión de la Universidad de Guayaquil ·
  <a href="https://admision.ug.edu.ec/admision/" target="_blank">admision.ug.edu.ec</a>
</div>
'''

with gr.Blocks(title='Chatbot UG - Admisión y Nivelación') as demo:
    gr.HTML(header_html)

    chatbot = gr.Chatbot(
        value=[],
        height=440,
        layout='bubble',
        watermark='Pregúntame sobre requisitos, fechas, carreras, costos, nivelación…',
        avatar_images=(USER_AVATAR_PATH, BOT_AVATAR_PATH),
        render_markdown=True,
        sanitize_html=True,
        line_breaks=True,
        group_consecutive_messages=True,
        feedback_options=('Like', 'Dislike'),
        elem_classes='chatbot',
    )
    estado = gr.Markdown('Escribe tu consulta para comenzar.', elem_id='estado')

    chips_state = gr.State(CHIPS_INICIALES)

    with gr.Row():
        with gr.Column(scale=0, min_width=210, elem_id='chips-col'):
            gr.HTML('<div class="chips-title">Preguntas sugeridas</div>')
            chip0 = gr.Button(CHIPS_INICIALES[0], elem_classes='chip')
            chip1 = gr.Button(CHIPS_INICIALES[1], elem_classes='chip')
            chip2 = gr.Button(CHIPS_INICIALES[2], elem_classes='chip')
            chip3 = gr.Button(CHIPS_INICIALES[3], elem_classes='chip')
        with gr.Column(scale=1):
            msg = gr.Textbox(
                placeholder='Escribe tu pregunta sobre admisión, requisitos, fechas, costos…',
                container=False, autofocus=True, elem_id='msg',
            )
        with gr.Column(scale=0, min_width=140, elem_id='btn-col'):
            enviar = gr.Button('Enviar', variant='primary', elem_id='enviar')
            limpiar_btn = gr.Button('Limpiar conversación', variant='secondary', elem_id='limpiar')

    browser_state = gr.BrowserState(default_value=None, storage_key='ug_chat_hist')

    gr.HTML(footer_html)

    salidas = [chatbot, estado, msg, chip0, chip1, chip2, chip3, chips_state, browser_state]

    msg.submit(handle_message, inputs=[msg, chatbot, browser_state], outputs=salidas)
    enviar.click(handle_message, inputs=[msg, chatbot, browser_state], outputs=salidas)
    chip0.click(partial(enviar_chip, 0), inputs=[chips_state, chatbot, browser_state], outputs=salidas)
    chip1.click(partial(enviar_chip, 1), inputs=[chips_state, chatbot, browser_state], outputs=salidas)
    chip2.click(partial(enviar_chip, 2), inputs=[chips_state, chatbot, browser_state], outputs=salidas)
    chip3.click(partial(enviar_chip, 3), inputs=[chips_state, chatbot, browser_state], outputs=salidas)

    limpiar_btn.click(limpiar_conversacion, outputs=[chatbot, estado, chip0, chip1, chip2, chip3, chips_state, browser_state])

    chatbot.like(registrar_feedback, inputs=[chatbot], outputs=None)

    demo.load(restaurar, inputs=browser_state, outputs=chatbot)
    demo.load(sugerencias_aleatorias, outputs=[chip0, chip1, chip2, chip3, chips_state])


if __name__ == '__main__':
    demo.launch(theme=theme, css=CSS)
