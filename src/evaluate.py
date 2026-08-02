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

test_queries = [
    # --- SALUDO / DESPEDIDA ---
    ('que tal todo', 'saludo'),
    ('buenas noches', 'saludo'),
    ('hola buenas que tal', 'saludo'),
    ('hey que mas', 'saludo'),
    ('hasta luego', 'despedida'),
    ('nos vemos luego', 'despedida'),
    ('gracias adios', 'despedida'),

    # --- REGISTRO NACIONAL ---
    ('donde hago el registro del minedec', 'registro_nacional'),
    ('me registre en minedec', 'registro_nacional'),
    ('como hago el registro nacional', 'registro_nacional'),

    # --- INSCRIPCION / POSTULACION ---
    ('como hago para postular a la ug', 'inscripcion_postulacion'),
    ('cuando abren las inscripciones', 'inscripcion_postulacion'),
    ('me quiero postular a la ug', 'inscripcion_postulacion'),

    # --- CRONOGRAMA ---
    ('cuando publican las fechas del examen', 'cronograma_evaluaciones'),
    ('cuando es el examen de ingreso', 'cronograma_evaluaciones'),
    ('donde veo el cronograma de evaluaciones', 'cronograma_evaluaciones'),

    # --- EVALUACION ADMISION ---
    ('de que trata la prueba de ingreso', 'evaluacion_admision'),
    ('en que consiste la prueba', 'evaluacion_admision'),
    ('como es el examen de admision', 'evaluacion_admision'),

    # --- ASIGNACION CUPOS ---
    ('como eligen a quien le dan cupo', 'asignacion_cupos'),
    ('como reparten los cupos', 'asignacion_cupos'),
    ('que nota necesito para entrar', 'asignacion_cupos'),
    ('puntaje minimo para obtener cupo', 'asignacion_cupos'),

    # --- ACEPTACION CUPO ---
    ('que hago si no me dieron cupo', 'aceptacion_cupo'),
    ('no sali seleccionado que hago', 'aceptacion_cupo'),
    ('como saber si me dieron cupo', 'aceptacion_cupo'),

    # --- REQUISITOS NIVELACION ---
    ('que documentos debo presentar para el curso de nivelacion', 'requisitos_nivelacion'),
    ('que llevo para la nivelacion', 'requisitos_nivelacion'),
    ('cuales son los requisitos de nivelacion', 'requisitos_nivelacion'),

    # --- MATRICULA NIVELACION ---
    ('como me inscribo en el curso de nivelacion', 'matricula_nivelacion'),
    ('donde me matriculo en nivelacion', 'matricula_nivelacion'),
    ('cuanto pago por el curso de nivelacion', 'matricula_nivelacion'),
    ('cuanto cuesta la nivelacion', 'matricula_nivelacion'),

    # --- EVALUACION NIVELACION ---
    ('que nota debo sacar para aprobar nivelacion', 'evaluacion_nivelacion'),
    ('como apruebo nivelacion', 'evaluacion_nivelacion'),
    ('como me evaluan en nivelacion', 'evaluacion_nivelacion'),

    # --- PLATAFORMAS VIRTUALES ---
    ('como entro al moodle de la universidad', 'plataformas_virtuales'),
    ('como entro a las clases virtuales', 'plataformas_virtuales'),

    # --- OFERTA ACADEMICA ---
    ('que carreras puedo estudiar en la ug', 'oferta_academica'),
    ('ke careras ug', 'oferta_academica'),
    ('cuantas carreras hay', 'oferta_academica'),

    # --- COSTOS / ARANCELES ---
    ('cuanto dinero debo pagar por credito', 'costos_aranceles'),
    ('precio de la matricula', 'costos_aranceles'),
    ('cuanto debo pagar por la carrera', 'costos_aranceles'),
    ('cuanto cuesta estudiar en la ug', 'costos_aranceles'),

    # --- GRATUIDAD ---
    ('la universidad es gratis', 'gratuidad_ug'),
    ('la ug es gratuita', 'gratuidad_ug'),
    ('la ug es gratis', 'gratuidad_ug'),

    # --- PERDIDA GRATUIDAD ---
    ('que pasa si repruebo materias y pierdo la gratuidad', 'perdida_gratuidad'),
    ('si repruebo tengo que pagar', 'perdida_gratuidad'),
    ('perdi la gratuidad que hago', 'perdida_gratuidad'),

    # --- DESCUENTOS VULNERABLES ---
    ('tengo descuento por discapacidad', 'descuentos_vulnerables'),
    ('cuanto pagan las personas con discapacidad', 'descuentos_vulnerables'),
    ('descuento adultos mayores aranceles', 'descuentos_vulnerables'),

    # --- REEMBOLSO ARANCELES ---
    ('reembolso arancel ug', 'reembolso_aranceles'),
    ('me devuelven el dinero si me retiro', 'reembolso_aranceles'),
    ('devolucion de valores pagados', 'reembolso_aranceles'),

    # --- TIPOS MATRICULA ---
    ('que diferencia hay entre matricula ordinaria y extraordinaria', 'tipos_matricula'),
    ('matricula tardia', 'tipos_matricula'),
    ('prorroga de matricula', 'tipos_matricula'),

    # --- TRANSFERENCIA ---
    ('vengo de otra universidad y quiero cambiarme a la ug', 'transferencia_estudiantes'),
    ('me quiero cambiar de carrera', 'transferencia_estudiantes'),
    ('homologar materias de otra universidad', 'transferencia_estudiantes'),

    # --- REQUISITOS EXTRANJEROS ---
    ('soy de colombia puedo estudiar en la ug', 'requisitos_extranjeros'),
    ('soy venezolano puedo entrar', 'requisitos_extranjeros'),
    ('soy extranjero puedo estudiar', 'requisitos_extranjeros'),

    # --- POLITICAS CUPOS ---
    ('hay cupos para personas con discapacidad', 'politicas_cupos'),
    ('hay cupos para discapacidad', 'politicas_cupos'),
    ('que es la accion afirmativa', 'politicas_cupos'),

    # --- CONTACTO ---
    ('cual es el telefono de la universidad', 'contacto'),
    ('cual es el numero de telefono', 'contacto'),

    # --- FALLBACK: vacios y basura ---
    ('', 'fallback'),
    ('xdxdddd', 'fallback'),

    # --- FALLBACK: OOD simple ---
    ('receta de arroz con pollo', 'fallback'),
    ('cuentame un chiste', 'fallback'),
    ('que hora es', 'fallback'),

    # --- FALLBACK: OOD adversarial ---
    ('noticias de la universidad de guayaquil hoy', 'fallback'),
    ('cuanto cuesta un taxi para la ug', 'fallback'),
    ('cuanto cuesta el pasaje de bus', 'fallback'),
    ('donde queda el mcdonalds', 'fallback'),
    ('que pelicula me recomiendas', 'fallback'),
    ('prestame dinero', 'fallback'),
    ('como esta el clima en guayaquil', 'fallback'),
]


def clasificar(agent, consulta):
    if not consulta or not consulta.strip():
        return 'fallback'
    consulta_proc = limpiar_texto(consulta, filtro_guayaquil=True)
    if not consulta_proc:
        return 'fallback'
    return agent.classifier.detectar_intencion(consulta_proc)[0]


def main():
    vec = Vectorizer()
    agent = Agent(vec)

    y_true, y_pred = [], []
    resultados = []
    for consulta, tag_real in test_queries:
        tag_pred = clasificar(agent, consulta)
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
    print(f'EVALUACION AMPLIADA - {len(test_queries)} CONSULTAS')
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
    plt.title('Matriz de Confusion - Evaluacion Ampliada', fontsize=14, pad=20)
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
1. Vocabulario fuera de dominio (OOD): El intent fallback ahora tiene
   ejemplos negativos, pero consultas OOD con palabras del dominio
   ("universidad", "cuesta") aun pueden colarse si se parecen mucho a un intent.

2. Dependencia de patrones: Solo se reconocen intenciones con patrones
   entrenados. Parafrasis muy alejadas pueden caer en fallback.

3. Sin contexto multi-turno: El agente no mantiene estado entre turnos.

4. Sin comprension semantica: TF-IDF compara terminos, no significado.
   Sinonimos no entrenados pueden fallar.
''')


if __name__ == '__main__':
    main()
