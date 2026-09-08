"""Entrena el modelo de scoring y guarda el artefacto junto con sus predicciones.

    python -m src.entrenar
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import datos
from src.config import (
    OBJETIVO,
    PROPORCION_PRUEBA,
    SEMILLA,
    VARIABLES_BINARIAS,
    VARIABLES_NUMERICAS,
)

SALIDA = Path(__file__).resolve().parents[1] / "artefactos"


def entrenar() -> dict:
    df = datos.construir_variables(datos.cargar())

    columnas = VARIABLES_NUMERICAS + VARIABLES_BINARIAS
    X = df[columnas]
    y = df[OBJETIVO]

    X_ent, X_pru, y_ent, y_pru, idx_ent, idx_pru = train_test_split(
        X, y, df.index, test_size=PROPORCION_PRUEBA,
        random_state=SEMILLA, stratify=y,
    )

    modelo = Pipeline(
        [
            ("escalado", StandardScaler()),
            ("clasificador", LogisticRegression(max_iter=2000, random_state=SEMILLA)),
        ]
    )
    modelo.fit(X_ent, y_ent)

    prob = modelo.predict_proba(X_pru)[:, 1]

    # El score de solvencia es el percentil del solicitante dentro de la
    # poblacion de referencia (la usada en entrenamiento). Es la practica
    # habitual en scoring crediticio: convierte una probabilidad concentrada
    # en un score repartido de forma uniforme entre 0 y 1, de modo que el
    # umbral de decision tenga un significado directo de negocio.
    prob_ref = np.sort(modelo.predict_proba(X_ent)[:, 1])
    score = 1.0 - np.searchsorted(prob_ref, prob, side="left") / len(prob_ref)

    SALIDA.mkdir(exist_ok=True)
    joblib.dump(modelo, SALIDA / "modelo.joblib")

    # Se guardan las predicciones del conjunto de prueba para que la evaluacion
    # sea un paso independiente del entrenamiento.
    pd.DataFrame(
        {
            "id_solicitud": df.loc[idx_pru, "id_solicitud"].values,
            "probabilidad_incumplimiento": prob,
            "score_solvencia": score,
            "incumplio": y_pru.values,
            "es_mujer_emprendedora": df.loc[idx_pru, "es_mujer_emprendedora"].values,
            "canal": df.loc[idx_pru, "canal"].values,
        }
    ).to_csv(SALIDA / "predicciones.csv", index=False)

    ficha = {
        "entrenado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filas_entrenamiento": int(len(X_ent)),
        "filas_prueba": int(len(X_pru)),
        "variables": columnas,
        "semilla": SEMILLA,
        "coeficientes": {
            col: round(float(c), 4)
            for col, c in zip(columnas, modelo.named_steps["clasificador"].coef_[0])
        },
    }
    (SALIDA / "ficha_modelo.json").write_text(
        json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return ficha


if __name__ == "__main__":
    ficha = entrenar()
    print("ENTRENAMIENTO COMPLETADO")
    print(f"  Filas de entrenamiento : {ficha['filas_entrenamiento']}")
    print(f"  Filas de prueba        : {ficha['filas_prueba']}")
    print(f"  Artefactos en          : artefactos/")
