"""Evalua el modelo contra las puertas de calidad y decide si puede desplegarse.

Termina con codigo de salida 1 si alguna puerta falla, lo que detiene el
pipeline y bloquea el despliegue. Ninguna persona interviene en esa decision.

    python -m src.evaluar
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import PUERTAS_DE_CALIDAD, UMBRAL_DECISION

ARTEFACTOS = Path(__file__).resolve().parents[1] / "artefactos"


def calcular_metricas(pred: pd.DataFrame, umbral: float) -> dict:
    """Metricas de negocio derivadas del umbral de decision."""
    aprobado = pred["score_solvencia"] >= umbral
    incumplio = pred["incumplio"] == 1

    rechazados_incumplidores = (~aprobado & incumplio).sum()
    total_incumplidores = incumplio.sum()

    tasa_mujeres = aprobado[pred["es_mujer_emprendedora"] == 1].mean()
    tasa_resto = aprobado[pred["es_mujer_emprendedora"] == 0].mean()

    return {
        # El AUC mide la capacidad de ordenar el riesgo:
        # es independiente del umbral de decision.
        "auc": float(roc_auc_score(pred["incumplio"], pred["probabilidad_incumplimiento"])),
        "recall_incumplidores": float(rechazados_incumplidores / total_incumplidores),
        "tasa_aprobacion": float(aprobado.mean()),
        "brecha_entre_segmentos": float(abs(tasa_mujeres - tasa_resto)),
        "tasa_aprobacion_mujeres": float(tasa_mujeres),
        "tasa_aprobacion_resto": float(tasa_resto),
    }


def evaluar_puertas(m: dict) -> list[dict]:
    """Aplica cada puerta de calidad. Devuelve el detalle de todas."""
    p = PUERTAS_DE_CALIDAD
    return [
        {
            "puerta": "AUC (capacidad de ordenar el riesgo)",
            "valor": m["auc"],
            "criterio": f">= {p['auc_minimo']:.2f}",
            "aprueba": m["auc"] >= p["auc_minimo"],
            "nota": "no depende del umbral",
        },
        {
            "puerta": "Recall de incumplidores",
            "valor": m["recall_incumplidores"],
            "criterio": f">= {p['recall_incumplidores_minimo']:.2f}",
            "aprueba": m["recall_incumplidores"] >= p["recall_incumplidores_minimo"],
            "nota": "protege la morosidad de la cartera",
        },
        {
            "puerta": "Tasa de aprobacion",
            "valor": m["tasa_aprobacion"],
            "criterio": (
                f"entre {p['tasa_aprobacion_minima']:.2f} "
                f"y {p['tasa_aprobacion_maxima']:.2f}"
            ),
            "aprueba": (
                p["tasa_aprobacion_minima"]
                <= m["tasa_aprobacion"]
                <= p["tasa_aprobacion_maxima"]
            ),
            "nota": "banda tolerada por el negocio",
        },
        {
            "puerta": "Brecha entre segmentos",
            "valor": m["brecha_entre_segmentos"],
            "criterio": f"<= {p['brecha_maxima_entre_segmentos']:.2f}",
            "aprueba": (
                m["brecha_entre_segmentos"] <= p["brecha_maxima_entre_segmentos"]
            ),
            "nota": "mujeres emprendedoras vs resto",
        },
    ]


def _linea(r: dict) -> str:
    marca = "PASA " if r["aprueba"] else "FALLA"
    return f"  [{marca}]  {r['puerta']:<38} {r['valor']:.4f}   ({r['criterio']})"


def _resumen_markdown(resultados: list[dict], aprobado: bool, m: dict) -> str:
    filas = "\n".join(
        f"| {'PASA' if r['aprueba'] else '**FALLA**'} | {r['puerta']} | "
        f"`{r['valor']:.4f}` | {r['criterio']} | {r['nota']} |"
        for r in resultados
    )
    veredicto = (
        "## Modelo APROBADO — puede desplegarse\n"
        if aprobado
        else "## Modelo RECHAZADO — el despliegue queda bloqueado\n"
    )
    return (
        f"{veredicto}\n"
        f"Umbral de decision evaluado: **{UMBRAL_DECISION:.2f}**\n\n"
        "| Resultado | Puerta de calidad | Valor | Criterio | |\n"
        "|---|---|---|---|---|\n"
        f"{filas}\n\n"
        f"Tasa de aprobacion en mujeres emprendedoras: "
        f"{m['tasa_aprobacion_mujeres']:.1%} · resto: {m['tasa_aprobacion_resto']:.1%}\n"
    )


def main() -> int:
    pred = pd.read_csv(ARTEFACTOS / "predicciones.csv")
    metricas = calcular_metricas(pred, UMBRAL_DECISION)
    resultados = evaluar_puertas(metricas)
    aprobado = all(r["aprueba"] for r in resultados)

    print("=" * 72)
    print(f"  PUERTAS DE CALIDAD  ·  umbral de decision = {UMBRAL_DECISION:.2f}")
    print("=" * 72)
    for r in resultados:
        print(_linea(r))
    print("-" * 72)
    print(
        "  VEREDICTO: MODELO APROBADO"
        if aprobado
        else "  VEREDICTO: MODELO RECHAZADO  ·  despliegue bloqueado"
    )
    print("=" * 72)

    (ARTEFACTOS / "reporte_calidad.json").write_text(
        json.dumps(
            {
                "umbral_decision": UMBRAL_DECISION,
                "aprobado": aprobado,
                "metricas": metricas,
                "puertas": resultados,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Publica el reporte en la pagina de resumen de GitHub Actions
    resumen = os.environ.get("GITHUB_STEP_SUMMARY")
    if resumen:
        with open(resumen, "a", encoding="utf-8") as fh:
            fh.write(_resumen_markdown(resultados, aprobado, metricas))

    return 0 if aprobado else 1


if __name__ == "__main__":
    sys.exit(main())
