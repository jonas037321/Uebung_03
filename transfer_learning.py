"""
Übung 03 - Kapitel 9: Transfer Learning mit MobileNetV2

MobileNetV2 wurde auf ImageNet (1,3 Mio. Bilder, 1000 Klassen) vortrainiert und
kennt daher schon Kanten, Texturen, Formen und Gegenstandsteile. Wir verwenden
den Faltungsteil als festen Feature Extractor und trainieren nur einen neuen
Klassifikationskopf für unsere Klassen:

    Input 128x128x3 (0..255)
    [Augmentation]                   wie in Kapitel 7
    Rescaling 1/127.5, -1            MobileNetV2 erwartet Pixel im Bereich -1..1
    MobileNetV2 (eingefroren)        ImageNet-Gewichte, ohne den alten 1000-Klassen-Kopf
    GlobalAveragePooling             1280 Merkmale
    Dropout 0.3
    Dense <Klassen>, Softmax         einziger trainierter Teil

Optional (--finetuning) werden danach die obersten Schichten von MobileNetV2
freigegeben und mit einer sehr kleinen Lernrate weitertrainiert.

Ergebnisse:  out/transfer/ bzw. out/transfer_finetuning/, Modelle in out/modelle/,
             Vergleich mit dem eigenen CNN in out/vergleich.md

Aufruf:
    python transfer_learning.py
    python transfer_learning.py --finetuning
"""

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
from keras import layers

from load_data import load_datasets
from training import INPUT_SHAPE, augmentation, count_params, evaluate_and_save, fit

FINE_TUNE_LAYERS = 30     # so viele oberste Schichten von MobileNetV2 werden beim Fine-Tuning freigegeben


def build_transfer(n_classes, flip=True, weights="imagenet"):
    base = keras.applications.MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, weights=weights)
    base.trainable = False
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = augmentation(flip)(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base(x, training=False)          # BatchNorm-Statistiken von ImageNet beibehalten
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="transfer"), base


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning mit MobileNetV2 (Kapitel 9)")
    parser.add_argument("--finetuning", action="store_true", help="danach obere Schichten nachtrainieren")
    parser.add_argument("--ohne-spiegeln", action="store_true", help="kein horizontales Spiegeln")
    parser.add_argument("--epochen", type=int, default=30, help="maximale Epochen pro Phase")
    parser.add_argument("--ohne-imagenet", action="store_true", help=argparse.SUPPRESS)  # nur zum Testen offline
    args = parser.parse_args()

    train_ds, val_ds, test_ds, class_names = load_datasets()
    model, base = build_transfer(len(class_names), not args.ohne_spiegeln,
                                 None if args.ohne_imagenet else "imagenet")
    model.summary()
    datasets = {"train": train_ds, "validation": val_ds, "test": test_ds}
    val_c = val_ds.cache()   # train_ds nicht cachen, sonst wäre die Reihenfolge jede Epoche gleich

    # Phase 1: nur der neue Kopf lernt
    hist, seconds = fit(model, train_ds, val_c, args.epochen, patience=8, learning_rate=1e-3)
    head_params, _ = count_params(model)
    evaluate_and_save("transfer", model, hist, seconds, datasets, class_names,
                      extra={"basismodell": "MobileNetV2 (ImageNet)", "eingefroren": True})
    if not args.finetuning:
        return

    # Phase 2 (optional): oberste Schichten freigeben, sehr kleine Lernrate
    print(f"\nFine-Tuning: die obersten {FINE_TUNE_LAYERS} Schichten von MobileNetV2 werden freigegeben ...")
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in base.layers:                       # BatchNorm bleibt eingefroren (kleiner Datensatz)
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    first_phase = len(hist["loss"])
    hist2, seconds2 = fit(model, train_ds, val_c, args.epochen, patience=8, learning_rate=1e-5, history=hist)
    evaluate_and_save("transfer_finetuning", model, hist2, seconds + seconds2, datasets, class_names,
                      fine_tune_start=first_phase,
                      extra={"basismodell": "MobileNetV2 (ImageNet)", "freigegebene_schichten": FINE_TUNE_LAYERS,
                             "trainierbare_parameter_phase1": head_params})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
