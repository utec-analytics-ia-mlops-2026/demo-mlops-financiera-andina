"""
Configuracion del modelo de scoring crediticio de Financiera Andina.

Este es el UNICO archivo que se modifica durante la demostracion en clase.
Todo lo demas del pipeline se ejecuta solo.
"""

# ---------------------------------------------------------------------------
# DECISION DE NEGOCIO
# ---------------------------------------------------------------------------
# El modelo estima la probabilidad de que un solicitante incumpla.
# Se APRUEBA la solicitud cuando esa probabilidad es MENOR al umbral.
#
# Subir el umbral  -> se aprueba a mas gente  -> se detectan menos incumplidores
# Bajar el umbral  -> se aprueba a menos gente -> se rechaza a mas clientes buenos
#
#   >>> DEMO EN CLASE: cambiar 0.50 por 0.60 y hacer push. <<<
#
UMBRAL_DECISION = 0.50

# ---------------------------------------------------------------------------
# PUERTAS DE CALIDAD
# ---------------------------------------------------------------------------
# Reglas escritas ANTES de entrenar. El pipeline las evalua solo y decide,
# sin intervencion humana, si el modelo puede pasar a produccion.
#
# Definidas por el Comite de Riesgos. Cambiarlas requiere acta del comite:
# no se tocan para que "pase" un modelo.
#
PUERTAS_DE_CALIDAD = {
    # Capacidad de ordenar el riesgo. NO depende del umbral de decision.
    "auc_minimo": 0.70,

    # De cada 100 clientes que efectivamente incumplieron, cuantos rechazamos.
    # Es la puerta que protege la morosidad de la cartera.
    "recall_incumplidores_minimo": 0.65,

    # Banda de aprobacion tolerada por el negocio.
    # Por debajo: dejamos de colocar. Por encima: asumimos riesgo no presupuestado.
    "tasa_aprobacion_minima": 0.42,
    "tasa_aprobacion_maxima": 0.62,

    # Equidad: diferencia maxima de tasa de aprobacion entre segmentos.
    # El 61% de la cartera son mujeres emprendedoras.
    "brecha_maxima_entre_segmentos": 0.15,
}

# ---------------------------------------------------------------------------
# PARAMETROS DEL ENTRENAMIENTO
# ---------------------------------------------------------------------------
SEMILLA = 2024
PROPORCION_PRUEBA = 0.30

VARIABLES_NUMERICAS = [
    "edad",
    "antiguedad_negocio_anios",
    "ingreso_mensual",
    "monto_solicitado",
    "deuda_actual",
    "ratio_endeudamiento",
    "ratio_cuota_ingreso",
]
VARIABLES_BINARIAS = ["tiene_historial_crediticio", "es_digital"]
OBJETIVO = "incumplio"

# ---------------------------------------------------------------------------
# CONTRATO DE DATOS
# ---------------------------------------------------------------------------
# Lo que el pipeline exige de los datos de entrada antes de entrenar.
CONTRATO_DATOS = {
    "columnas_obligatorias": [
        "id_solicitud",
        "edad",
        "antiguedad_negocio_anios",
        "ingreso_mensual",
        "monto_solicitado",
        "deuda_actual",
        "tiene_historial_crediticio",
        "canal",
        "region",
        "es_mujer_emprendedora",
        "incumplio",
    ],
    "filas_minimas": 3000,
    "porcentaje_nulos_maximo": 0.02,
    "rangos": {
        "edad": (18, 85),
        "antiguedad_negocio_anios": (0, 60),
        # Rango en SOLES. Si la fuente cambiara a dolares, esta puerta lo detecta.
        "ingreso_mensual": (400, 60000),
        "monto_solicitado": (500, 150000),
        "deuda_actual": (0, 200000),
    },
    "categorias": {
        "canal": {"presencial", "digital"},
        "region": {"lima", "arequipa", "trujillo"},
    },
    "tasa_incumplimiento_esperada": (0.05, 0.25),
}
