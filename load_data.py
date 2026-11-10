"""
Übung 03 - Kapitel 4: Einlesen der Daten mit Keras

Liest data/train, data/validation und data/test mit
keras.utils.image_dataset_from_directory() ein und bringt alle Bilder auf
128 x 128 Pixel. Die Funktion load_datasets() wird in den späteren Kapiteln
(eigenes CNN, Augmentation, Transfer Learning) wiederverwendet.

Beim direkten Aufruf werden Klassen und Bildanzahlen ausgegeben und
out/beispielbilder.png (je Bereich ein paar Beispiele) gespeichert.

Aufruf:
    python load_data.py
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TensorFlow-Infomeldungen ausblenden

import keras
import numpy as np

from config import BATCH_SIZE, DATA_DIR, IMG_SIZE, OUT_DIR, SEED, SPLITS


def load_datasets(data_dir=DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Gibt (train_ds, val_ds, test_ds, class_names) zurück.

    Pixelwerte bleiben 0..255 (float32); das Normalisieren übernimmt später
    eine Rescaling-Schicht im Modell, damit Vorhersage und Training garantiert
    gleich vorverarbeiten.
    """
    for split in SPLITS:
        if not (data_dir / split).exists():
            raise SystemExit(f"{data_dir / split} fehlt - zuerst split_data.py ausführen!")

    def load(split, shuffle):
        return keras.utils.image_dataset_from_directory(
            data_dir / split,
            labels="inferred",
            label_mode="int",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=SEED,
            verbose=False,
        )

    train_ds = load("train", shuffle=True)
    val_ds = load("validation", shuffle=False)
    test_ds = load("test", shuffle=False)

    class_names = train_ds.class_names
    for name, ds in (("validation", val_ds), ("test", test_ds)):
        if ds.class_names != class_names:
            raise SystemExit(f"Klassen in '{name}' ({ds.class_names}) passen nicht zu "
                             f"train ({class_names}) - Aufteilung prüfen!")
    return train_ds, val_ds, test_ds, class_names


def count_per_class(ds, n_classes):
    counts = np.zeros(n_classes, dtype=int)
    for _, labels in ds:
        counts += np.bincount(labels.numpy(), minlength=n_classes)
    return counts


def save_examples(datasets, class_names, path, per_split=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(datasets), per_split, figsize=(2 * per_split, 2.3 * len(datasets)))
    for row, (split, ds) in enumerate(datasets.items()):
        images, labels = next(iter(ds.unbatch().shuffle(500, seed=SEED).batch(per_split)))
        for col in range(per_split):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(images):
                ax.imshow(images[col].numpy().astype("uint8"))
                ax.set_title(class_names[int(labels[col])], fontsize=9)
        axes[row, 0].text(-0.15, 0.5, split, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=11, weight="bold")
    fig.suptitle(f"Beispielbilder nach dem Einlesen ({IMG_SIZE[0]}x{IMG_SIZE[1]})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    train_ds, val_ds, test_ds, class_names = load_datasets()
    datasets = {"train": train_ds, "validation": val_ds, "test": test_ds}

    images, labels = next(iter(train_ds))
    print(f"Klassen ({len(class_names)}): {class_names}")
    print(f"Form eines Batches: Bilder {tuple(images.shape)}, Labels {tuple(labels.shape)}, "
          f"Wertebereich {float(images.numpy().min()):.0f}..{float(images.numpy().max()):.0f}")
    print()
    print(f"{'Klasse':<16}" + "".join(f"{s:>12}" for s in datasets))
    print("-" * (16 + 12 * len(datasets)))
    counts = {s: count_per_class(ds, len(class_names)) for s, ds in datasets.items()}
    for i, name in enumerate(class_names):
        print(f"{name:<16}" + "".join(f"{counts[s][i]:>12}" for s in datasets))
    print("-" * (16 + 12 * len(datasets)))
    print(f"{'Summe':<16}" + "".join(f"{counts[s].sum():>12}" for s in datasets))

    path = OUT_DIR / "beispielbilder.png"
    save_examples(datasets, class_names, path)
    print(f"\nBeispielbilder gespeichert: {path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
