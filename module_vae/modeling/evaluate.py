"""Avaliação final no conjunto de teste e simulação da fila de triagem (Seção 8)."""

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def metrics_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """Sensibilidade, especificidade, precisão e matriz de confusão em um limiar."""
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "precision": tp / max(tp + fp, 1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_model(y_test: np.ndarray, score_test: np.ndarray, threshold: float) -> dict:
    """Avaliação final do score no conjunto de teste.

    - AUC-ROC: separação geral, independente de limiar.
    - AP (precisão média): mais informativa com classes desbalanceadas. A AP de um score
      aleatório é a proporção de positivos (62,5% no teste), não 0,5.
    - Sensibilidade e especificidade no limiar calibrado na validação.
    """
    results = {
        "roc_auc": float(roc_auc_score(y_test, score_test)),
        "average_precision": float(average_precision_score(y_test, score_test)),
        "threshold": threshold,
        **metrics_at_threshold(y_test, score_test, threshold),
    }
    results["sensitivity"] = float(results["sensitivity"])
    results["specificity"] = float(results["specificity"])
    results["precision"] = float(results["precision"])

    logger.info("=" * 60)
    logger.success("AVALIAÇÃO NO TESTE")
    logger.info(f"AUC-ROC: {results['roc_auc']:.3f}")
    logger.info(f"AP: {results['average_precision']:.3f} (acaso: {y_test.mean():.3f})")
    logger.info(f"Sensibilidade: {results['sensitivity']:.3f}")
    logger.info(f"Especificidade: {results['specificity']:.3f}")
    logger.info(f"Precisão (VPP): {results['precision']:.3f}")
    logger.info(f"Matriz de confusão: {results['confusion_matrix']}")

    return results


def compare_thresholds(
    thresholds: dict[str, float],
    y_val: np.ndarray,
    score_val: np.ndarray,
    y_test: np.ndarray,
    score_test: np.ndarray,
) -> pd.DataFrame:
    """Compara a sensibilidade e a especificidade de vários limiares na validação e no teste.

    Se a especificidade do teste ficar bem abaixo da validação, o "normal" mudou entre os
    conjuntos (Seção 9.3 do notebook) e o limiar precisaria ser recalibrado.
    """
    rows = []
    for name, threshold in thresholds.items():
        for split, y, score in [("val", y_val, score_val), ("test", y_test, score_test)]:
            m = metrics_at_threshold(y, score, threshold)
            rows.append(
                {
                    "threshold": name,
                    "split": split,
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                    "precision": m["precision"],
                }
            )

    table = pd.DataFrame(rows).set_index(["threshold", "split"])
    logger.info(f"Limiares de decisão:\n{table.round(3).to_string()}")

    return table


def simulate_queue(
    y: np.ndarray,
    score: np.ndarray,
    n_queues: int = 2000,
    queue_size: int = 100,
    prevalence: float = 0.10,
    seed: int = 0,
) -> dict:
    """Simula filas realistas e compara a ordem de chegada com a ordem pelo score.

    O teste tem 62,5% de pneumonias, muito acima de uma fila de pronto-atendimento. Aqui
    sorteamos filas de queue_size exames com a prevalência dada e medimos onde as
    pneumonias ficam em cada ordenação.

    Returns:
        dict: Posição média de uma pneumonia na fila (1 a queue_size) e quantas
            pneumonias aparecem entre os 10 primeiros, nas duas ordens.
    """
    rng = np.random.default_rng(seed)
    normal_idx, pneumonia_idx = np.where(y == 0)[0], np.where(y == 1)[0]
    n_pneumonia = round(queue_size * prevalence)

    pos_arrival, pos_model, top10_arrival, top10_model = [], [], [], []
    for _ in range(n_queues):
        queue = np.concatenate(
            [
                rng.choice(normal_idx, queue_size - n_pneumonia, replace=False),
                rng.choice(pneumonia_idx, n_pneumonia, replace=False),
            ]
        )
        rng.shuffle(queue)  # ordem de chegada aleatória
        queue_model = queue[np.argsort(-score[queue])]  # ordem pelo score do VAE

        pos_arrival.append((np.where(y[queue] == 1)[0] + 1).mean())
        pos_model.append((np.where(y[queue_model] == 1)[0] + 1).mean())
        top10_arrival.append(y[queue[:10]].sum())
        top10_model.append(y[queue_model[:10]].sum())

    results = {
        "mean_position_arrival": float(np.mean(pos_arrival)),
        "mean_position_model": float(np.mean(pos_model)),
        "top10_arrival": float(np.mean(top10_arrival)),
        "top10_model": float(np.mean(top10_model)),
    }

    logger.success(f"FILA SIMULADA ({queue_size} exames, {prevalence:.0%} de pneumonias)")
    logger.info(
        "Posição média de uma pneumonia: "
        f"{results['mean_position_arrival']:.0f} → {results['mean_position_model']:.0f}"
    )
    logger.info(
        f"Pneumonias entre os 10 primeiros: {results['top10_arrival']:.1f} → "
        f"{results['top10_model']:.1f}"
    )

    return results
