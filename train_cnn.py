"""
Übung 03 - Kapitel 5 + 7: eigenes CNN trainieren (ohne / mit Data Augmentation)

Architektur (selbst entworfen, 128 x 128 x 3 Eingabe):

    Input 128x128x3
    [Augmentation]                        nur mit --augmentation (Kapitel 7)
    Rescaling 1/255                       Pixel 0..255 -> 0..1 (im Modell, damit predict.py gleich vorverarbeitet)
    Conv2D 16, 3x3, ReLU  -> MaxPooling   128 -> 64   Kanten, Farbübergänge
    Conv2D 32, 3x3, ReLU  -> MaxPooling    64 -> 32   einfache Formen
    Conv2D 64, 3x3, ReLU  -> MaxPooling    32 -> 16   Teile von Gegenständen
    Conv2D 128, 3x3, ReLU                  16 x 16    größere Strukturen
    GlobalAveragePooling                   128 Werte  statt Flatten: viel weniger Parameter
    Dropout 0.3
    Dense 64, ReLU
    Dense <Klassen>, Softmax

Vier Faltungsblöcke, weil die Gegenstände bei 128 Pixeln groß sind: Nach drei
Poolings deckt ein Neuron der letzten Faltung ~1/3 des Bildes ab. GlobalAveragePooling
statt Flatten hält das Modell klein (~106.000 statt über 2 Millionen Parameter),
was bei nur ~70 Trainingsbildern pro Klasse Overfitting verringert.

Ergebnisse:  out/cnn/ bzw. out/cnn_augmentation/  (Lernkurven, Konfusionsmatrix,
             Beispielbilder, ergebnisse.json), Modell in out/modelle/, Vergleich in out/vergleich.md

Aufruf:
    python train_cnn.py                    # Kapitel 5: eigenes CNN
    python train_cnn.py --augmentation     # Kapitel 7: mit Data Augmentation
    python train_cnn.py --augmentation --ohne-spiegeln   # falls Spiegeln für eure Gegenstände keinen Sinn ergibt
"""

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
from keras import layers

from load_data import load_datasets
from training import INPUT_SHAPE, augmentation, evaluate_and_save, fit


def build_cnn(n_classes, augment=False, flip=True):
    return keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),
            *([augmentation(flip)] if augment else []),   # vor Rescaling: RandomBrightness erwartet 0..255
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="cnn_augmentation" if augment else "cnn",
    )


def main():
    parser = argparse.ArgumentParser(description="Eigenes CNN trainieren (Kapitel 5 / 7)")
    parser.add_argument("--augmentation", action="store_true", help="Data Augmentation einbauen (Kapitel 7)")
    parser.add_argument("--ohne-spiegeln", action="store_true", help="kein horizontales Spiegeln")
    parser.add_argument("--epochen", type=int, default=None,
                        help="maximale Epochen (Standard: 60 ohne, 120 mit Augmentation)")
    args = parser.parse_args()

    train_ds, val_ds, test_ds, class_names = load_datasets()
    model = build_cnn(len(class_names), args.augmentation, not args.ohne_spiegeln)
    model.summary()

    # Augmentation macht jede Epoche schwieriger -> mehr Epochen und mehr Geduld
    epochs = args.epochen or (120 if args.augmentation else 60)
    patience = 20 if args.augmentation else 10
    hist, seconds = fit(model, train_ds, val_ds.cache(), epochs, patience)

    evaluate_and_save(model.name, model, hist, seconds,
                      {"train": train_ds, "validation": val_ds, "test": test_ds}, class_names,
                      extra={"augmentation": args.augmentation, "spiegeln": not args.ohne_spiegeln})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
