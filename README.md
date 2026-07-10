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
│   └── intents.json              → 21 intents con utterances y respuestas
├── 📁 src/
│   ├── __init__.py
│   ├── preprocess.py             → Limpieza, tokenización, stopwords, stemming
│   ├── vectorizer.py             → TF-IDF (unigramas + bigramas)
│   ├── classifier.py             → Similitud coseno + umbral de confianza + fallback
│   ├── entity_extractor.py       → Extracción de entidades (fechas, cédulas, carreras)
│   ├── agent.py                  → Lógica central del chatbot
│   ├── app.py                    → Interfaz web con Gradio
│   └── evaluate.py               → Evaluación con 28 consultas de prueba
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
| `python src/evaluate.py` | Evaluación del agente (accuracy, F1, matriz de confusión) |

## Fuentes de datos reales

- [Portal Oficial de Admisión UG](https://admision.ug.edu.ec/admision/)
- [Página de Nivelación UG](https://admision.ug.edu.ec/nivelacion/)
- [Oferta Académica UG](https://admision.ug.edu.ec/oferta-ug/)
- Blog alaU — FAQ sobre ingreso a la UG
- Reglamento de Matrículas, Aranceles y Derechos UG 2023 (Reformado)
- Reglamento de Admisión de la UG 2023 (Reformado)

## Funcionalidades

- **RF-01:** 21 intents con utterances y respuestas en `data/intents.json`
- **RF-02:** Preprocesamiento con stopwords (NLTK) y stemming (Snowball)
- **RF-03:** Representación TF-IDF con unigramas y bigramas
- **RF-04:** Detección de intenciones por similitud coseno
- **RF-05:** Extracción de entidades con regex (fechas, cédulas, carreras, montos)
- **RF-06:** Umbral de confianza (0.35) con respuesta de fallback
- **RF-07:** Interfaz web con Gradio
- **RF-08:** Evaluación con accuracy y F1-macro
