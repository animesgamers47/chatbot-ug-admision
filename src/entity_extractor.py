import re


CARRERAS_UG = [
    'administracion de empresas', 'agronomia', 'agropecuaria', 'alimentos',
    'arquitectura', 'biologia', 'bioquimica y farmacia',
    'ciencia de datos e inteligencia artificial', 'ciencias politicas',
    'comercio exterior', 'comunicacion', 'contabilidad y auditoria',
    'derecho', 'diseno de interiores', 'diseno grafico', 'economia',
    'economia internacional', 'educacion basica', 'educacion inicial',
    'enfermeria', 'entrenamiento deportivo', 'finanzas', 'fonoaudiologia',
    'gastronomia', 'geologia', 'gestion de la informacion gerencial',
    'ingenieria ambiental', 'ingenieria civil', 'ingenieria de la produccion',
    'ingenieria industrial', 'ingenieria quimica', 'mecatronica', 'medicina',
    'medicina veterinaria', 'mercadotecnia', 'negocios internacionales',
    'nutricion y dietetica', 'obstetricia', 'odontologia',
    'pedagogia de la actividad fisica y deporte', 'pedagogia de la filosofia',
    'pedagogia de la historia y ciencias sociales', 'pedagogia de la informatica',
    'pedagogia de la lengua y literatura',
    'pedagogia de las matematicas y la fisica',
    'pedagogia de la quimica y biologia',
    'pedagogia de las artes y las humanidades',
    'pedagogia de los idiomas nacionales y extranjeros', 'psicologia',
    'psicologia educativa', 'publicidad', 'sistemas de informacion',
    'sociologia', 'software', 'tecnologias de la informacion', 'telematica',
    'terapia ocupacional', 'terapia respiratoria', 'turismo'
]


def extraer_entidades(texto):
    entidades = {}
    t = texto.lower()

    fechas = re.findall(r'\b\d{1,2}\s+de\s+[a-záéíóú]+(?:\s+del?\s+\d{4})?\b', t)
    if fechas:
        entidades['fechas'] = fechas

    cedulas = re.findall(r'\b\d{10}\b', texto)
    if cedulas:
        entidades['cédulas'] = cedulas

    montos = re.findall(r'\$\s*\d+(?:\.\d{2})?', texto)
    if montos:
        entidades['montos'] = montos

    porcentajes = re.findall(r'\b\d+\s*%\b', texto)
    if porcentajes:
        entidades['porcentajes'] = porcentajes

    carreras_detectadas = []
    for carrera in CARRERAS_UG:
        if carrera in t:
            carreras_detectadas.append(carrera.title())
    if carreras_detectadas:
        entidades['carreras'] = carreras_detectadas

    procesos_ug = [
        'curso de nivelacion', 'cupo aceptado', 'cupo asignado',
        'matricula ordinaria', 'matricula extraordinaria', 'matricula especial',
        'registro nacional', 'inscripcion', 'postulacion',
        'evaluacion de admision', 'examen de admision',
        'aceptacion de cupo', 'perdida de gratuidad',
        'accion afirmativa', 'requisitos de nivelacion'
    ]
    procesos_detectados = []
    for proceso in procesos_ug:
        if proceso in t:
            procesos_detectados.append(proceso.title())
    if procesos_detectados:
        entidades['procesos'] = procesos_detectados

    return entidades if entidades else None
