"""CAPA 2 — PRUEBAS DE DATOS.

Verifican que los datos de entrada cumplan el contrato antes de entrenar.
Esta capa NO existe en el software tradicional, y es donde se cuelan los
errores mas caros: el codigo funciona, el modelo entrena sin fallar, y las
predicciones son sistematicamente absurdas.
"""

import pandas as pd
import pytest

from src.config import CONTRATO_DATOS
from src.datos import cargar, validar


@pytest.fixture(scope="module")
def df():
    return cargar()


def test_los_datos_cumplen_el_contrato(df):
    fallos = validar(df)
    assert fallos == [], "Incumplimientos del contrato de datos:\n" + "\n".join(fallos)


def test_hay_volumen_suficiente(df):
    assert len(df) >= CONTRATO_DATOS["filas_minimas"]


def test_no_hay_identificadores_duplicados(df):
    assert df["id_solicitud"].is_unique


def test_el_objetivo_es_binario(df):
    assert set(df["incumplio"].unique()) <= {0, 1}


def test_no_hay_ingresos_ni_montos_negativos(df):
    assert (df["ingreso_mensual"] > 0).all()
    assert (df["monto_solicitado"] > 0).all()
    assert (df["deuda_actual"] >= 0).all()


def test_la_unidad_monetaria_sigue_siendo_soles(df):
    """Detecta el fallo clasico: la fuente cambia de soles a dolares.

    El codigo seguiria funcionando y el modelo entrenaria sin errores.
    Solo una prueba de datos lo atrapa.
    """
    mediana = df["ingreso_mensual"].median()
    assert 1200 <= mediana <= 12000, (
        f"Mediana de ingreso mensual = {mediana:.0f}. "
        "Fuera del rango esperado en soles: revisar si la fuente cambio de moneda."
    )


def test_la_tasa_de_incumplimiento_es_plausible(df):
    minimo, maximo = CONTRATO_DATOS["tasa_incumplimiento_esperada"]
    tasa = df["incumplio"].mean()
    assert minimo <= tasa <= maximo, f"Tasa de incumplimiento anomala: {tasa:.1%}"


def test_ambos_canales_estan_representados(df):
    proporciones = df["canal"].value_counts(normalize=True)
    assert proporciones.min() >= 0.10, "Un canal quedo subrepresentado en los datos"


def test_el_validador_detecta_una_columna_faltante(df):
    mutilado = df.drop(columns=["ingreso_mensual"])
    assert any("Faltan columnas" in f for f in validar(mutilado))


def test_el_validador_detecta_un_cambio_de_moneda(df):
    """Simula que la fuente empieza a entregar los ingresos en dolares."""
    en_dolares = df.copy()
    en_dolares["ingreso_mensual"] = en_dolares["ingreso_mensual"] / 3.75
    fallos = validar(en_dolares)
    assert any("ingreso_mensual" in f for f in fallos), (
        "El contrato de datos deberia detectar el cambio de unidad monetaria"
    )


def test_el_validador_detecta_una_categoria_nueva(df):
    con_region_nueva = df.copy()
    con_region_nueva.loc[con_region_nueva.index[:50], "region"] = "cusco"
    fallos = validar(con_region_nueva)
    assert any("categorias no esperadas" in f for f in fallos), (
        "Entrar a una region nueva debe obligar a revisar el modelo, no pasar inadvertido"
    )
