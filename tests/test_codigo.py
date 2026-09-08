"""CAPA 1 — PRUEBAS DE CODIGO.

Verifican que las funciones hagan lo que dicen. Es la capa que tambien
existe en cualquier proyecto de software tradicional.
"""

import pandas as pd
import pytest

from src.config import PUERTAS_DE_CALIDAD, UMBRAL_DECISION
from src.datos import construir_variables
from src.evaluar import calcular_metricas


def _muestra():
    return pd.DataFrame(
        {
            "ingreso_mensual": [1000.0, 2000.0, 4000.0],
            "deuda_actual": [500.0, 0.0, 8000.0],
            "monto_solicitado": [12000.0, 24000.0, 48000.0],
            "canal": ["digital", "presencial", "digital"],
        }
    )


def test_ratio_endeudamiento_se_calcula_bien():
    df = construir_variables(_muestra())
    assert df["ratio_endeudamiento"].tolist() == [0.5, 0.0, 2.0]


def test_ratio_cuota_ingreso_se_calcula_bien():
    df = construir_variables(_muestra())
    assert df["ratio_cuota_ingreso"].round(4).tolist() == [1.0, 1.0, 1.0]


def test_canal_digital_se_codifica_como_binaria():
    df = construir_variables(_muestra())
    assert df["es_digital"].tolist() == [1, 0, 1]


def test_construir_variables_no_muta_el_dataframe_original():
    original = _muestra()
    copia = original.copy()
    construir_variables(original)
    pd.testing.assert_frame_equal(original, copia)


def test_umbral_de_decision_esta_en_rango_valido():
    assert 0.0 < UMBRAL_DECISION < 1.0, "El umbral debe ser una fraccion entre 0 y 1"


def test_la_banda_de_aprobacion_es_coherente():
    p = PUERTAS_DE_CALIDAD
    assert p["tasa_aprobacion_minima"] < p["tasa_aprobacion_maxima"]


@pytest.mark.parametrize("umbral", [0.30, 0.50, 0.70])
def test_metricas_devuelve_valores_en_rango(umbral):
    pred = pd.DataFrame(
        {
            "score_solvencia": [0.9, 0.7, 0.4, 0.2, 0.6, 0.1],
            "probabilidad_incumplimiento": [0.05, 0.2, 0.5, 0.8, 0.3, 0.9],
            "incumplio": [0, 0, 1, 1, 0, 1],
            "es_mujer_emprendedora": [1, 0, 1, 0, 1, 0],
        }
    )
    m = calcular_metricas(pred, umbral)
    for clave in ("auc", "recall_incumplidores", "tasa_aprobacion"):
        assert 0.0 <= m[clave] <= 1.0, f"{clave} fuera de rango"


def test_umbral_mas_alto_no_puede_aprobar_a_mas_gente():
    """Propiedad del score: subir el umbral solo puede reducir aprobaciones."""
    pred = pd.DataFrame(
        {
            "score_solvencia": [0.9, 0.7, 0.4, 0.2, 0.6, 0.1],
            "probabilidad_incumplimiento": [0.05, 0.2, 0.5, 0.8, 0.3, 0.9],
            "incumplio": [0, 0, 1, 1, 0, 1],
            "es_mujer_emprendedora": [1, 0, 1, 0, 1, 0],
        }
    )
    laxo = calcular_metricas(pred, 0.30)["tasa_aprobacion"]
    estricto = calcular_metricas(pred, 0.70)["tasa_aprobacion"]
    assert estricto <= laxo
