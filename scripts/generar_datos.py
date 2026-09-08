"""
Genera el dataset sintetico de solicitudes de microcredito de Financiera Andina.

Solo hace falta ejecutarlo si se quiere regenerar data/solicitudes.csv.
El CSV ya viene incluido en el repositorio, asi que en la demo NO se ejecuta.

    python scripts/generar_datos.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42
N = 6000
SALIDA = Path(__file__).resolve().parents[1] / "data" / "solicitudes.csv"


def generar(n: int = N, semilla: int = SEMILLA) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    canal = rng.choice(["presencial", "digital"], size=n, p=[0.66, 0.34])
    es_digital = (canal == "digital").astype(int)

    region = rng.choice(
        ["lima", "arequipa", "trujillo"], size=n, p=[0.55, 0.25, 0.20]
    )
    # El canal digital se lanzo con foco en mujeres emprendedoras,
    # asi que el segmento esta sobrerrepresentado ahi.
    es_mujer_emprendedora = rng.binomial(
        1, np.where(es_digital == 1, 0.68, 0.57), size=n
    )

    # El canal digital trae clientes mas jovenes y con menos huella crediticia
    edad = np.clip(
        rng.normal(43, 11, size=n) - es_digital * 11, 19, 78
    ).round().astype(int)

    antiguedad_negocio = np.clip(
        rng.gamma(shape=2.6, scale=2.4, size=n) - es_digital * 1.6, 0.2, 35
    ).round(1)

    ingreso_mensual = np.clip(
        rng.lognormal(mean=8.25, sigma=0.52, size=n) * (1 - 0.30 * es_digital),
        450, 30000,
    ).round(0)

    # Menos huella crediticia en el canal digital y en el sector mas informal
    prob_historial = np.where(es_digital == 1, 0.51, 0.82) - 0.09 * es_mujer_emprendedora
    tiene_historial = rng.binomial(1, np.clip(prob_historial, 0.05, 0.95), size=n)

    monto_solicitado = np.clip(
        ingreso_mensual * rng.uniform(1.4, 4.5, size=n), 800, 90000
    ).round(0)

    deuda_actual = np.clip(
        ingreso_mensual * rng.gamma(shape=1.3, scale=0.9, size=n), 0, 120000
    ).round(0)

    ratio_endeudamiento = np.clip(deuda_actual / ingreso_mensual, 0, 25)
    ratio_cuota = np.clip(monto_solicitado / (ingreso_mensual * 12), 0, 6)

    # Modelo generador del incumplimiento (log-odds)
    log_odds = (
        -2.45
        - 0.022 * (edad - 40)
        - 0.115 * (antiguedad_negocio - 5)
        - 0.68 * tiene_historial
        + 0.34 * ratio_endeudamiento
        + 0.55 * ratio_cuota
        + 0.58 * es_digital
        - 0.28 * np.log(ingreso_mensual / 3000)
        + rng.normal(0, 0.55, size=n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    incumplio = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "id_solicitud": [f"SOL-{i:06d}" for i in range(1, n + 1)],
            "edad": edad,
            "antiguedad_negocio_anios": antiguedad_negocio,
            "ingreso_mensual": ingreso_mensual,
            "monto_solicitado": monto_solicitado,
            "deuda_actual": deuda_actual,
            "tiene_historial_crediticio": tiene_historial,
            "canal": canal,
            "region": region,
            "es_mujer_emprendedora": es_mujer_emprendedora,
            "incumplio": incumplio,
        }
    )
    return df


if __name__ == "__main__":
    df = generar()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA, index=False)
    print(f"Escrito {SALIDA}  ·  {len(df)} filas")
    print(f"Tasa de incumplimiento: {df['incumplio'].mean():.3f}")
    print(df.groupby('canal')['incumplio'].mean().round(3).to_string())
