import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from src.vectorizer import Vectorizer
from src.agent import Agent
from src.preprocess import limpiar_texto

vec = Vectorizer()
agent = Agent(vec)

test_queries = [
    # --- SALUDO (reformulaciones) ---
    ('que tal todo', 'saludo'),
    ('buenas noches', 'saludo'),

    # --- REGISTRO NACIONAL ---
    ('donde hago el registro del minedec', 'registro_nacional'),

    # --- INSCRIPCION / POSTULACION ---
    ('como hago para postular a la ug', 'inscripcion_postulacion'),

    # --- CRONOGRAMA ---
    ('cuando publican las fechas del examen', 'cronograma_evaluaciones'),

    # --- EVALUACION ADMISION ---
    ('de que trata la prueba de ingreso', 'evaluacion_admision'),

    # --- ASIGNACION CUPOS ---
    ('como eligen a quien le dan cupo', 'asignacion_cupos'),

    # --- ACEPTACION CUPO ---
    ('que hago si no me dieron cupo', 'aceptacion_cupo'),

    # --- REQUISITOS NIVELACION ---
    ('que documentos debo presentar para el curso de nivelacion', 'requisitos_nivelacion'),

    # --- MATRICULA NIVELACION ---
    ('como me inscribo en el curso de nivelacion', 'matricula_nivelacion'),

    # --- EVALUACION NIVELACION ---
    ('que nota debo sacar para aprobar nivelacion', 'evaluacion_nivelacion'),

    # --- PLATAFORMAS VIRTUALES ---
    ('como entro al moodle de la universidad', 'plataformas_virtuales'),

    # --- OFERTA ACADEMICA ---
    ('que carreras puedo estudiar en la ug', 'oferta_academica'),
    ('ke careras ug', 'oferta_academica'),

    # --- COSTOS / ARANCELES ---
    ('cuanto dinero debo pagar por credito', 'costos_aranceles'),

    # --- GRATUIDAD ---
    ('la universidad es gratis', 'gratuidad_ug'),

    # --- PERDIDA GRATUIDAD ---
    ('que pasa si repruebo materias y pierdo la gratuidad', 'perdida_gratuidad'),

    # --- TIPOS MATRICULA ---
    ('que diferencia hay entre matricula ordinaria y extraordinaria', 'tipos_matricula'),

    # --- TRANSFERENCIA ---
    ('vengo de otra universidad y quiero cambiarme a la ug', 'transferencia_estudiantes'),

    # --- REQUISITOS EXTRANJEROS ---
    ('soy de colombia puedo estudiar en la ug', 'requisitos_extranjeros'),

    # --- POLITICAS CUPOS ---
    ('hay cupos para personas con discapacidad', 'politicas_cupos'),

    # --- CONTACTO ---
    ('cual es el telefono de la universidad', 'contacto'),

    # --- DESPEDIDA ---
    ('hasta luego', 'despedida'),

    # --- FALLBACK: OOD simple ---
    ('', 'fallback'),
    ('receta de arroz con pollo', 'fallback'),

    # --- FALLBACK: OOD adversarial ---
    ('noticias de la universidad de guayaquil hoy', 'fallback'),
    ('cuanto cuesta un taxi para la ug', 'fallback'),

    # --- FALLBACK: Typos ---
    ('xdxdddd', 'fallback'),
]

y_true, y_pred = [], []
resultados = []

for consulta, tag_real in test_queries:
    if not consulta or not consulta.strip():
        tag_pred = 'fallback'
    else:
        consulta_proc = limpiar_texto(consulta, filtro_guayaquil=True)
        if not consulta_proc:
            tag_pred = 'fallback'
        else:
            tag_pred, _ = agent.classifier.detectar_intencion(consulta_proc)

    y_true.append(tag_real)
    y_pred.append(tag_pred)
    resultados.append({
        'consulta': consulta,
        'real': tag_real,
        'pred': tag_pred,
        'correcto': 'OK' if tag_pred == tag_real else 'ERR'
    })

df_resultados = pd.DataFrame(resultados)

accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

print('=' * 60)
print('EVALUACION ADVERSARIAL - TEST DE GENERALIZACION')
print('=' * 60)
print(f'\nAccuracy:  {accuracy:.2%}')
print(f'F1-Macro:  {f1_macro:.2%}')
print(f'Correctas: {sum(1 for r in resultados if r["correcto"] == "OK")}/{len(resultados)}')

print('\nReporte de clasificacion:')
print(classification_report(y_true, y_pred, zero_division=0))

errores = df_resultados[df_resultados['correcto'] == 'ERR']
if len(errores) > 0:
    print(f'\nErrores ({len(errores)}):')
    for _, row in errores.iterrows():
        print(f'  "{row["consulta"]}" -> real: {row["real"]}, pred: {row["pred"]}')
else:
    print('\nSin errores!')

tags_unicos = sorted(set(y_true + y_pred))
cm = confusion_matrix(y_true, y_pred, labels=tags_unicos)

plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=tags_unicos, yticklabels=tags_unicos)
plt.title('Matriz de Confusion - Evaluacion Adversarial', fontsize=14, pad=20)
plt.xlabel('Prediccion', fontsize=12)
plt.ylabel('Real', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=150, bbox_inches='tight')
print('\nGrafico guardado: matriz_confusion.png')

print('\n' + '=' * 60)
print('LIMITACIONES OBSERVADAS')
print('=' * 60)
print('''
1. Vocabulario fuera de dominio (OOD): Las consultas que contienen
   palabras del dominio ("universidad", "estudiante", "cuesta") pero
   que no son sobre admision/nivelacion pueden clasificarse
   incorrectamente. Esto es una limitacion inherente de TF-IDF.

2. Dependencia de patrones: El sistema solo reconoce intenciones
   para las cuales tiene patrones de entrenamiento. Consultas
   formuladas de forma muy diferente a los patrones pueden no
   ser detectadas.

3. Falta de contexto: El agente no mantiene estado entre turnos,
   por lo que no puede manejar conversaciones multi-turno.

4. Sin comprension semantica: TF-IDF compara terminos, no
   significado. Sinonimos o parfrasis no entrenadas pueden
   fallar en la clasificacion.
''')

if __name__ == '__main__':
    pass
