"""Carga de datos, construccion de variables y validacion del contrato de datos."""

from pathlib import Path

import pandas as pd

from src.config import CONTRATO_DATOS

RUTA_DATOS = Path(__file__).resolve().parents[1] / "data" / "solicitudes.csv"


def cargar(ruta: Path | str = RUTA_DATOS) -> pd.DataFrame:
    """Lee el CSV de solicitudes."""
    return pd.read_csv(ruta)


def construir_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva las variables que consume el modelo.

    Se mantiene deliberadamente simple: la demo trata sobre automatizacion,
    no sobre ingenieria de variables.
    """
    df = df.copy()
    df["ratio_endeudamiento"] = df["deuda_actual"] / df["ingreso_mensual"]
    df["ratio_cuota_ingreso"] = df["monto_solicitado"] / (df["ingreso_mensual"] * 12)
    df["es_digital"] = (df["canal"] == "digital").astype(int)
    return df


def validar(df: pd.DataFrame) -> list[str]:
    """Comprueba el contrato de datos.

    Devuelve la lista de incumplimientos encontrados. Lista vacia = datos aptos.
    """
    fallos: list[str] = []
    c = CONTRATO_DATOS

    faltantes = set(c["columnas_obligatorias"]) - set(df.columns)
    if faltantes:
        fallos.append(f"Faltan columnas obligatorias: {sorted(faltantes)}")
        return fallos  # sin columnas no tiene sentido seguir validando

    if len(df) < c["filas_minimas"]:
        fallos.append(
            f"Volumen insuficiente: {len(df)} filas, minimo {c['filas_minimas']}"
        )

    for col in c["columnas_obligatorias"]:
        pct_nulos = df[col].isna().mean()
        if pct_nulos > c["porcentaje_nulos_maximo"]:
            fallos.append(
                f"Columna '{col}': {pct_nulos:.1%} de nulos "
                f"(maximo {c['porcentaje_nulos_maximo']:.1%})"
            )

    for col, (minimo, maximo) in c["rangos"].items():
        fuera = df[(df[col] < minimo) | (df[col] > maximo)]
        if len(fuera) > 0:
            fallos.append(
                f"Columna '{col}': {len(fuera)} valores fuera del rango "
                f"[{minimo}, {maximo}]"
            )

    for col, permitidas in c["categorias"].items():
        inesperadas = set(df[col].dropna().unique()) - permitidas
        if inesperadas:
            fallos.append(f"Columna '{col}': categorias no esperadas {sorted(inesperadas)}")

    tasa = df["incumplio"].mean()
    minimo, maximo = c["tasa_incumplimiento_esperada"]
    if not (minimo <= tasa <= maximo):
        fallos.append(
            f"Tasa de incumplimiento {tasa:.1%} fuera del rango esperado "
            f"[{minimo:.0%}, {maximo:.0%}]"
        )

    return fallos


if __name__ == "__main__":
    import sys

    df = cargar()
    fallos = validar(df)
    print(f"Filas leidas: {len(df)}")
    if fallos:
        print("\nVALIDACION DE DATOS: FALLIDA")
        for f in fallos:
            print(f"  - {f}")
        sys.exit(1)
    print("VALIDACION DE DATOS: OK")
