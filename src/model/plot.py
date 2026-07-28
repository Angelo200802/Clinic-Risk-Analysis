from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F
import numpy as np
import matplotlib.pyplot as plt
import json, os
 
def _compute_roc(scores, labels):
    """
    Calcola manualmente i punti (fpr, tpr) e l'AUC di una curva ROC.
 
    Nota: si usa questa implementazione invece di
    pyspark.mllib.evaluation.BinaryClassificationMetrics.roc() perché quel
    metodo esiste solo nell'API Scala/Java sottostante e NON è esposto dal
    wrapper Python (causa AttributeError: 'BinaryClassificationMetrics'
    object has no attribute 'roc').
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
 
    order = np.argsort(-scores)  # ordina per score decrescente
    labels_sorted = labels[order]
 
    P = labels_sorted.sum()          # totale positivi
    N = len(labels_sorted) - P       # totale negativi
 
    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)
 
    tpr = tps / P if P > 0 else np.zeros_like(tps)
    fpr = fps / N if N > 0 else np.zeros_like(fps)
 
    # aggiunge il punto iniziale (0,0)
    fpr = np.concatenate(([0.0], fpr))
    tpr = np.concatenate(([0.0], tpr))
 
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc
 
 
def plot_roc_curves(model_paths: dict, test_df: DataFrame, label_col: str = "RiskCategory_b",
                     save_path: str = None):
    """
    Args:
        model_paths: dizionario {nome_modello: path_del_pipeline_salvato}
                      es. {"logistic_regression": SAVE_MODEL_PATH+"/log_reg_pipeline",
                           "naive_bayes": SAVE_MODEL_PATH+"/nb_pipeline",
                           "mlp": SAVE_MODEL_PATH+"/ann_model_pipeline"}
                      Se contiene un solo modello, plotta una sola curva.
        test_df: DataFrame Spark di test, con le colonne ORIGINALI del dataset
                 (lo stesso "test" prodotto da ds.randomSplit(...) nei tuoi script,
                 prima di applicare qualsiasi pipeline: il PipelineModel si occupa
                 lui stesso di indexing/assembling/scaling).
        label_col: colonna target indicizzata usata in training (default "RiskCategory_b").
        save_path: se specificato, salva il grafico su file invece di mostrarlo a schermo.
    """
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(7, 7))
 
    for i, (model_name, path) in enumerate(model_paths.items()):
        print(f"Carico e valuto il modello '{model_name}' da: {path}")
        model = PipelineModel.load(path)
        predictions = model.transform(test_df)
 
        # Estrae la probabilità della classe positiva (indice 1 del vettore "probability")
        scored = (
            predictions
            .withColumn("score", vector_to_array("probability")[1])
            .select(F.col("score").cast("double"), F.col(label_col).cast("double").alias("label"))
        )
 
        rows = scored.collect()  # porta score/label sul driver (ok per dataset di dimensioni gestibili)
        scores = [r["score"] for r in rows]
        labels = [r["label"] for r in rows]
 
        fpr, tpr, auc = _compute_roc(scores, labels)
 
        color = colors[i % len(colors)]
        plt.plot(fpr, tpr, color=color, linewidth=2, label=f"{model_name} (AUC = {auc:.3f})")
 
    # Diagonale di riferimento (classificatore casuale)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUC = 0.5)")
 
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grafico salvato in: {save_path}")
    else:
        plt.show()


def plot_metrics_comparison(metrics_path: str, models: list = None,
                             metrics_to_plot: list = None, save_path: str = None):
    """
    Args:
        metrics_path: path del file metrics.json (o della cartella che lo contiene).
        models: lista di nomi dei modelli da confrontare. Se None, usa tutti quelli nel file.
        metrics_to_plot: lista delle metriche da mostrare (es. ["accuracy", "f1"]).
                          Se None, usa ["accuracy", "precision", "recall", "f1", "auc_roc"].
        save_path: se specificato, salva il grafico su file invece di mostrarlo.
    """
    if os.path.isdir(metrics_path):
        metrics_path = os.path.join(metrics_path, "metrics.json")
 
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"File non trovato: {metrics_path}")
 
    with open(metrics_path, "r") as f:
        all_metrics = json.load(f)
 
    if models is None:
        models = list(all_metrics.keys())
 
    if metrics_to_plot is None:
        metrics_to_plot = ["accuracy", "precision", "recall", "f1", "auc_roc"]
 
    # Filtra i modelli effettivamente presenti nel file
    models = [m for m in models if m in all_metrics]
    if not models:
        raise ValueError("Nessuno dei modelli richiesti è presente in metrics.json")
 
    n_models = len(models)
    n_metrics = len(metrics_to_plot)
 
    x = np.arange(n_metrics)          # una posizione per metrica
    width = 0.8 / n_models            # larghezza barra, per far stare tutti i modelli affiancati
 
    colors = plt.cm.tab10.colors
 
    plt.figure(figsize=(9, 6))
 
    for i, model_name in enumerate(models):
        values = [all_metrics[model_name].get(metric, 0.0) for metric in metrics_to_plot]
        offset = (i - (n_models - 1) / 2) * width
        bars = plt.bar(x + offset, values, width, label=model_name, color=colors[i % len(colors)])
 
        # etichetta numerica sopra ogni barra
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                      f"{value:.3f}", ha="center", va="bottom", fontsize=8)
 
    plt.xticks(x, metrics_to_plot)
    plt.ylabel("Valore")
    plt.ylim(0, 1.05)
    plt.title("Confronto metriche tra modelli")
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grafico salvato in: {save_path}")
    else:
        plt.show()