"""
Übung 03 - gemeinsame Bausteine für Training und Auswertung (Kapitel 5-7, 9)

Wird von train_cnn.py und transfer_learning.py verwendet:
  - augmentation()       Data-Augmentation-Block (Kapitel 7)
  - fit()                Training mit Early Stopping + Zeitmessung
  - evaluate_and_save()  Accuracy auf train/validation/test, Accuracy pro
                         Aufnahmeserie (Kapitel 6), Lernkurven, Konfusionsmatrix,
                         Beispielbilder richtig/falsch, Modell speichern
  - write_comparison()   out/vergleich.md mit allen bisher trainierten Varianten

Alle Ergebnisse einer Variante landen in out/<variante>/, die Modelle in out/modelle/.
"""

import json
import time
from pathlib import Path

import keras
import numpy as np
from keras import layers

from config import IMG_SIZE, OUT_DIR, SEED

MODEL_DIR = OUT_DIR / "modelle"
INPUT_SHAPE = (*IMG_SIZE, 3)

VARIANTS = {   # Reihenfolge und Beschriftung in der Vergleichstabelle
    "cnn": "Eigenes CNN",
    "cnn_augmentation": "Eigenes CNN + Data Augmentation",
    "transfer": "Transfer Learning (MobileNetV2, Feature Extractor)",
    "transfer_finetuning": "Transfer Learning + Fine-Tuning",
}


# --------------------------------------------------------------------------- Modellbausteine

def augmentation(flip=True):
    """Zufällige Veränderungen, nur während des Trainings aktiv (Kapitel 7)."""
    steps = [
        layers.RandomRotation(0.05),            # +/- 18 Grad
        layers.RandomTranslation(0.1, 0.1),     # bis 10 % in jede Richtung verschieben
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.15),          # etwas heller/dunkler (andere Beleuchtung)
        layers.RandomContrast(0.15),
    ]
    if flip:
        steps.insert(0, layers.RandomFlip("horizontal"))
    return keras.Sequential(steps, name="augmentation")


def count_params(model):
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    total = sum(int(np.prod(w.shape)) for w in model.weights)
    return trainable, total


def fit(model, train_ds, val_ds, epochs, patience=10, learning_rate=1e-3, history=None):
    """Kompiliert und trainiert; gibt (history_dict, sekunden) zurück.

    Early Stopping beobachtet val_loss und stellt am Ende die besten Gewichte wieder her.
    Mit history= kann ein bestehender Verlauf (z.B. vor dem Fine-Tuning) fortgesetzt werden.
    """
    keras.utils.set_random_seed(SEED)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # start_from_epoch: nicht abbrechen, bevor das Netz überhaupt richtig zu lernen begonnen hat
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, start_from_epoch=10,
                                         restore_best_weights=True, verbose=1)
    start = time.time()
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[stop], verbose=2)
    seconds = time.time() - start
    hist = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    if history:
        hist = {k: history.get(k, []) + hist[k] for k in hist}
    return hist, seconds


# --------------------------------------------------------------------------- Auswertung

def predict_all(model, ds):
    """Wahrscheinlichkeiten, echte Labels und Bilder (uint8) eines nicht gemischten Datasets."""
    probs, labels, images = [], [], []
    for x, y in ds:
        probs.append(model.predict_on_batch(x))
        labels.append(y.numpy())
        images.append(x.numpy().astype("uint8"))
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(images)


def series_group(path):
    """Aus 'serie_07__maus_serie_07_003.jpg' bzw. 'testserie_01__...' die Gruppe bestimmen."""
    series = Path(path).name.split("__", 1)[0]
    return "spätere Testbilder (testserie_*)" if series.startswith("testserie_") else f"{series} (gleiche Sitzung)"


def plot_history(hist, title, path, fine_tune_start=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(hist["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, name in ((ax1, "loss", "Loss"), (ax2, "accuracy", "Accuracy")):
        ax.plot(epochs, hist[key], "o-", ms=3, label="Training")
        ax.plot(epochs, hist[f"val_{key}"], "o-", ms=3, label="Validierung")
        if fine_tune_start:
            ax.axvline(fine_tune_start + 0.5, color="grey", ls="--", label="Start Fine-Tuning")
        ax.set_xlabel("Epoche")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend()
    best = int(np.argmin(hist["val_loss"])) + 1
    ax1.axvline(best, color="green", ls=":", alpha=0.7)
    ax1.annotate("beste Epoche", (best, max(hist["val_loss"])), color="green", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_confusion(y_true, y_pred, class_names, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(1.3 * n + 2, 1.3 * n + 1.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(n), class_names, rotation=30, ha="right")
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("vorhergesagt")
    ax.set_ylabel("tatsächlich")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return cm


def plot_examples(images, y_true, probs, paths, class_names, title, path, n_each=8):
    """Je eine Zeile korrekte und falsche Vorhersagen (spätere Testbilder bevorzugt)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_pred = probs.argmax(1)
    later = np.array(["testserie_" in Path(p).name for p in paths])
    rng = np.random.default_rng(SEED)

    def pick(mask):
        first = rng.permutation(np.flatnonzero(mask & later))
        rest = rng.permutation(np.flatnonzero(mask & ~later))
        return np.concatenate([first, rest])[:n_each].astype(int)

    rows = [("richtig", pick(y_pred == y_true)), ("falsch", pick(y_pred != y_true))]
    fig, axes = plt.subplots(2, n_each, figsize=(2 * n_each, 6))
    for r, (name, idx) in enumerate(rows):
        for c in range(n_each):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(idx):
                if c == 0:
                    ax.text(0.5, 0.5, "keine", ha="center", va="center", transform=ax.transAxes)
                continue
            i = idx[c]
            ax.imshow(images[i])
            color = "green" if y_pred[i] == y_true[i] else "red"
            ax.set_title(f"ist: {class_names[y_true[i]]}\nerkannt: {class_names[y_pred[i]]} "
                         f"({probs[i, y_pred[i]]:.0%})\n{Path(paths[i]).name.split('__')[0]}",
                         fontsize=7.5, color=color)
        axes[r, 0].text(-0.12, 0.5, name, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=11, weight="bold")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def overfitting_text(hist):
    best = int(np.argmin(hist["val_loss"]))
    gap = hist["accuracy"][best] - hist["val_accuracy"][best]
    last = len(hist["val_loss"]) - 1
    rising = hist["val_loss"][last] > hist["val_loss"][best] * 1.15 and last - best >= 3
    if hist["accuracy"][best] < 0.6:
        return (f"Underfitting: Schon auf den Trainingsdaten nur {hist['accuracy'][best]:.0%} Accuracy - "
                f"das Modell hat (noch) kaum etwas gelernt. Mehr Epochen (--epochen) oder bessere Daten versuchen.")
    if gap > 0.15 or rising:
        reasons = []
        if gap > 0.05:
            reasons.append(f"in der besten Epoche ({best + 1}) liegt die Trainings-Accuracy "
                           f"{gap:.0%} über der Validierungs-Accuracy")
        if rising:
            reasons.append(f"der Validierungs-Loss steigt nach Epoche {best + 1} wieder an, "
                           f"während der Trainings-Loss weiter fällt")
        return "Anzeichen für Overfitting: " + " und ".join(reasons) + "."
    if gap > 0.05:
        return (f"Leichtes Overfitting: Trainings- und Validierungs-Accuracy liegen in der besten "
                f"Epoche ({best + 1}) {gap:.0%} auseinander.")
    return (f"Kaum Overfitting: Trainings- und Validierungskurven verlaufen nahe beieinander "
            f"(Abstand {abs(gap):.0%} in Epoche {best + 1}).")


def evaluate_and_save(name, model, hist, seconds, datasets, class_names, fine_tune_start=None, extra=None):
    """Wertet ein trainiertes Modell aus und speichert alles unter out/<name>/."""
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    title = VARIANTS.get(name, name)

    model_path = MODEL_DIR / f"{name}.keras"
    model.save(model_path)
    (MODEL_DIR / "klassen.json").write_text(json.dumps(class_names, ensure_ascii=False), encoding="utf-8")

    lines = []
    model.summary(print_fn=lambda line, **_: lines.append(line))
    (out / "architektur.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nAuswertung ...")
    acc = {}
    for split in ("train", "validation", "test"):
        _, acc[split] = model.evaluate(datasets[split], verbose=0)

    # Kapitel 6: Test-Accuracy getrennt nach Herkunft der Testbilder
    test_ds = datasets["test"]
    probs, y_true, images = predict_all(model, test_ds)
    y_pred = probs.argmax(1)
    paths = test_ds.file_paths
    groups = {}
    for g in sorted({series_group(p) for p in paths}, key=lambda s: s.startswith("spät")):
        mask = np.array([series_group(p) == g for p in paths])
        groups[g] = {"accuracy": float((y_pred[mask] == y_true[mask]).mean()), "bilder": int(mask.sum())}

    # Klassen, die bei den späteren Testbildern am häufigsten verwechselt werden
    wrong = [{"datei": Path(p).name, "ist": class_names[t], "erkannt": class_names[q],
              "wahrscheinlichkeit": round(float(pr[q]), 3)}
             for p, t, q, pr in zip(paths, y_true, y_pred, probs) if t != q]
    wrong.sort(key=lambda w: -w["wahrscheinlichkeit"])

    plot_history(hist, f"{title} - Lernkurven", out / "lernkurven.png", fine_tune_start)
    cm = plot_confusion(y_true, y_pred, class_names, f"{title} - Testdaten", out / "konfusionsmatrix.png")
    plot_examples(images, y_true, probs, paths, class_names,
                  f"{title} - Testbilder (grün = richtig, rot = falsch)", out / "beispiele_richtig_falsch.png")

    trainable, total = count_params(model)
    best = int(np.argmin(hist["val_loss"]))
    result = {
        "variante": name,
        "titel": title,
        "trainierbare_parameter": trainable,
        "parameter_gesamt": total,
        "epochen": len(hist["loss"]),
        "beste_epoche": best + 1,
        "trainingszeit_s": round(seconds, 1),
        "accuracy": {k: round(float(v), 4) for k, v in acc.items()},
        "test_nach_herkunft": groups,
        "konfusionsmatrix": cm.tolist(),
        "overfitting": overfitting_text(hist),
        "falsch_klassifiziert": wrong[:30],
        **(extra or {}),
    }
    (out / "ergebnisse.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "verlauf.json").write_text(json.dumps(hist), encoding="utf-8")

    print(f"\n{title}")
    print(f"  trainierbare Parameter: {trainable:,}  (gesamt {total:,})")
    print(f"  Training: {len(hist['loss'])} Epochen, beste Epoche {best + 1}, {seconds:.0f} s")
    print(f"  Accuracy  train {acc['train']:.1%}   validation {acc['validation']:.1%}   test {acc['test']:.1%}")
    for g, v in groups.items():
        print(f"    test, {g:<36} {v['accuracy']:.1%}  ({v['bilder']} Bilder)")
    print(f"  {result['overfitting']}")
    print(f"  Modell: {model_path}")
    print(f"  Plots:  {out}")
    write_comparison()
    return result


# --------------------------------------------------------------------------- Vergleich

def write_comparison():
    """Sammelt alle out/<variante>/ergebnisse.json in out/vergleich.md."""
    results = []
    for name in VARIANTS:
        f = OUT_DIR / name / "ergebnisse.json"
        if f.exists():
            results.append(json.loads(f.read_text(encoding="utf-8")))
    if not results:
        return
    groups = []
    for r in results:
        for g in r["test_nach_herkunft"]:
            if g not in groups:
                groups.append(g)

    md = ["# Übung 03 - Vergleich der Modelle", "",
          "Automatisch erzeugt von `training.py` nach jedem Training.", "",
          "| Variante | trainierbare Parameter | Epochen | Trainingszeit | Acc. train | Acc. validation | Acc. test | "
          + " | ".join(f"Test: {g}" for g in groups) + " |",
          "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(groups)]
    for r in results:
        a = r["accuracy"]
        cells = [f"{r['test_nach_herkunft'][g]['accuracy']:.1%} ({r['test_nach_herkunft'][g]['bilder']})"
                 if g in r["test_nach_herkunft"] else "-" for g in groups]
        md.append(f"| {r['titel']} | {r['trainierbare_parameter']:,} | {r['epochen']} | "
                  f"{r['trainingszeit_s']:.0f} s | {a['train']:.1%} | {a['validation']:.1%} | {a['test']:.1%} | "
                  + " | ".join(cells) + " |")
    md += ["", "*Test: serie_07* = gemischte Serie aus derselben Aufnahmesitzung wie das Training; "
               "*spätere Testbilder* = an anderem Tag / Ort / mit anderem Exemplar aufgenommen "
               "(der eigentliche Generalisierungstest).", ""]
    for r in results:
        md += [f"## {r['titel']}", "",
               f"- Overfitting: {r['overfitting']}",
               f"- Lernkurven: [`{r['variante']}/lernkurven.png`]({r['variante']}/lernkurven.png)",
               f"- Konfusionsmatrix: [`{r['variante']}/konfusionsmatrix.png`]({r['variante']}/konfusionsmatrix.png)",
               f"- Beispiele: [`{r['variante']}/beispiele_richtig_falsch.png`]"
               f"({r['variante']}/beispiele_richtig_falsch.png)",
               f"- Architektur: [`{r['variante']}/architektur.txt`]({r['variante']}/architektur.txt)", ""]
        if r["falsch_klassifiziert"]:
            md += ["Sicherste Fehlklassifikationen im Test:", "",
                   "| Datei | ist | erkannt als | Wahrscheinlichkeit |", "|---|---|---|---:|"]
            md += [f"| {w['datei']} | {w['ist']} | {w['erkannt']} | {w['wahrscheinlichkeit']:.0%} |"
                   for w in r["falsch_klassifiziert"][:8]]
            md.append("")
    path = OUT_DIR / "vergleich.md"
    path.write_text("\n".join(md), encoding="utf-8")
    print(f"  Vergleich aktualisiert: {path}")
