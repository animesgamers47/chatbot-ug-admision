# Chatbot UG - Agente Conversacional para Admisión y Nivelación

**Universidad de Guayaquil** — Facultad de Ciencias Matemáticas y Físicas  
**Carrera de Ciencia de Datos & IA** — Procesamiento de Lenguaje Natural  
**Trabajo Parcial II**

---

## Objetivo

Agente conversacional basado en técnicas clásicas de PLN (TF-IDF + similitud coseno) que identifica la intención del usuario y responde preguntas frecuentes sobre admisión y nivelación de la Universidad de Guayaquil, utilizando datos reales extraídos de fuentes oficiales.

## Estructura del repositorio

```
📁 ProyectoPLN/
├── 📁 data/
│   └── intents.json              → 25 intents con utterances y respuestas
├── 📁 src/
│   ├── __init__.py
│   ├── preprocess.py             → Limpieza, tokenización, stopwords, stemming
│   ├── vectorizer.py             → TF-IDF (unigramas + bigramas)
│   ├── classifier.py             → Max-pooling por intent + umbral/margen + fallback
│   ├── entity_extractor.py       → Extracción de entidades (fechas, cédulas, carreras)
│   ├── agent.py                  → Lógica central del chatbot + consola ejecutable
│   ├── app.py                    → Interfaz web con Gradio
│   └── evaluate.py               → Evaluación con 83 consultas (incluye adversariales)
├── requirements.txt
└── README.md
```

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecución

| Comando | Descripción |
|---------|-------------|
| `python src/app.py` | Interfaz web tipo chat (Gradio) |
| `python src/agent.py` | Chat por consola (escribe "salir" para terminar) |
| `python src/evaluate.py` | Evaluación del agente (accuracy, F1, matriz de confusión) |

## Despliegue en línea

- **URL:** https://chatbot-ug-admision.onrender.com
- La app se despliega automáticamente desde la rama `v2-chatbot-ug` (plan free: duerme tras ~15 min de inactividad y despierta en 30–50 s).

## Fuentes de datos reales

- [Portal Oficial de Admisión UG](https://admision.ug.edu.ec/admision/)
- [Página de Nivelación UG](https://admision.ug.edu.ec/nivelacion/)
- [Oferta Académica UG](https://admision.ug.edu.ec/oferta-ug/)
- Blog alaU — FAQ sobre ingreso a la UG
- Reglamento de Matrículas, Aranceles y Derechos UG 2023 (Reformado)
- Reglamento de Admisión de la UG 2023 (Reformado)

## Funcionalidades

- **RF-01:** 25 intents con utterances y respuestas en `data/intents.json`
- **RF-02:** Preprocesamiento con stopwords (NLTK) y stemming (Snowball)
- **RF-03:** Representación TF-IDF con unigramas y bigramas
- **RF-04:** Detección de intenciones por similitud coseno con **max-pooling por intent** (el mejor puntaje de cada intent, no de cada utterance)
- **RF-05:** Extracción de entidades con regex (fechas, cédulas, carreras, montos) y enriquecimiento de la respuesta con las entidades detectadas
- **RF-06:** Umbral de confianza (0.40) **+ margen mínimo** (0.05) entre el mejor y el segundo intent para rechazar consultas fuera de tema
- **RF-07:** Intent `fallback` con patrones negativos (OOD) para aprender qué NO es tema del bot
- **RF-08:** Interfaz web con Gradio (preguntas sugeridas aleatorias, feedback 👍/👎 a `data/feedback.csv`, persistencia del historial y diseño responsive)
- **RF-09:** Consola ejecutable: `python src/agent.py`
- **RF-10:** Evaluación con 83 consultas (incluye reformulaciones y casos adversariales)
- **RF-11:** Despliegue en Render (https://chatbot-ug-admision.onrender.com) con auto-deploy

## Resultados de evaluación

`python src/evaluate.py` sobre un set ampliado de 83 consultas:

- **Accuracy:** 100%
- **F1-Macro:** 100%
- **Falsos positivos OOD:** 0

> Nota: el set incluye reformulaciones no vistas en entrenamiento, consultas de
> confusión entre intents similares (matrícula/costos, cupos/gratuidad) y
> consultas fuera de dominio (noticias, taxis, comida, clima, etc.).
