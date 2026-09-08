"""CAPA 3 — PRUEBAS DE MODELO.

Verifican el comportamiento del modelo entrenado: que sea reproducible, que
tenga sentido de negocio y que no se rompa con entradas extremas.

Se ejecutan DESPUES del entrenamiento, sobre el artefacto generado.
Son distintas de las puertas de calidad: aqui se prueba que el modelo se
comporte de forma sensata; alli se decide si puede ir a produccion.
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.config import VARIABLES_BINARIAS, VARIABLES_NUMERICAS

ARTEFACTOS = Path(__file__).resolve().parents[1] / "artefactos"

pytestmark = pytest.mark.skipif(
    not (ARTEFACTOS / "modelo.joblib").exists(),
    reason="No hay modelo entrenado: ejecutar primero 'python -m src.entrenar'",
)


@pytest.fixture(scope="module")
def modelo():
    return joblib.load(ARTEFACTOS / "modelo.joblib")


@pytest.fixture(scope="module")
def predicciones():
    return pd.read_csv(ARTEFACTOS / "predicciones.csv")


def _solicitante(**cambios):
    base = {
        "edad": 42,
        "antiguedad_negocio_anios": 6.0,
        "ingreso_mensual": 4000.0,
        "monto_solicitado": 12000.0,
        "deuda_actual": 3000.0,
        "ratio_endeudamiento": 0.75,
        "ratio_cuota_ingreso": 0.25,
        "tiene_historial_crediticio": 1,
        "es_digital": 0,
    }
    base.update(cambios)
    return pd.DataFrame([base])[VARIABLES_NUMERICAS + VARIABLES_BINARIAS]


def test_las_probabilidades_estan_entre_cero_y_uno(predicciones):
    p = predicciones["probabilidad_incumplimiento"]
    assert p.between(0, 1).all()


def test_el_score_esta_entre_cero_y_uno(predicciones):
    s = predicciones["score_solvencia"]
    assert s.between(0, 1).all()


def test_el_modelo_es_reproducible(modelo):
    """Dos llamadas con la misma entrada deben dar el mismo resultado."""
    entrada = _solicitante()
    assert modelo.predict_proba(entrada)[0, 1] == modelo.predict_proba(entrada)[0, 1]


def test_mas_endeudamiento_no_reduce_el_riesgo(modelo):
    """Prueba de sentido de negocio: la direccion del efecto debe ser la esperada."""
    poco = modelo.predict_proba(_solicitante(ratio_endeudamiento=0.2))[0, 1]
    mucho = modelo.predict_proba(_solicitante(ratio_endeudamiento=5.0))[0, 1]
    assert mucho > poco, "A mayor endeudamiento, el riesgo estimado deberia subir"


def test_tener_historial_crediticio_no_perjudica(modelo):
    con = modelo.predict_proba(_solicitante(tiene_historial_crediticio=1))[0, 1]
    sin = modelo.predict_proba(_solicitante(tiene_historial_crediticio=0))[0, 1]
    assert sin >= con, "No tener historial no deberia reducir el riesgo estimado"


def test_el_modelo_no_se_rompe_con_valores_extremos(modelo):
    extremo = _solicitante(
        edad=85, ingreso_mensual=450.0, deuda_actual=200000.0,
        ratio_endeudamiento=25.0, ratio_cuota_ingreso=6.0,
    )
    p = modelo.predict_proba(extremo)[0, 1]
    assert 0.0 <= p <= 1.0


def test_el_modelo_discrimina_entre_perfiles(predicciones):
    """Los que incumplieron deben tener, en promedio, peor score."""
    incumplidores = predicciones.loc[predicciones["incumplio"] == 1, "score_solvencia"]
    cumplidores = predicciones.loc[predicciones["incumplio"] == 0, "score_solvencia"]
    assert incumplidores.mean() < cumplidores.mean()


def test_ninguna_prediccion_quedo_vacia(predicciones):
    assert predicciones["probabilidad_incumplimiento"].notna().all()
    assert len(predicciones) > 0
