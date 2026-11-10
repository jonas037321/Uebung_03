"""
Übung 03 - Kapitel 8: Vorhersage mit neuen Bildern bzw. Kamera

Das Programm
  - lädt ein gespeichertes Modell aus out/modelle/,
  - liest neue Bilder (Dateien/Ordner) oder Kamera-Frames ein,
  - bereitet sie GENAU wie die Trainingsbilder vor: quadratischer Ausschnitt aus
    der Bildmitte (wie beim Aufnehmen), bilinear auf 128 x 128 skaliert (wie
    image_dataset_from_directory), Pixel 0..255 (das Normalisieren macht das Modell),
  - berechnet die Vorhersage und gibt Klasse + Wahrscheinlichkeit aus.

Ohne --modell wird das Modell mit der besten Test-Accuracy verwendet.

Aufruf:
    python predict.py bild.jpg                    # einzelnes Bild
    python predict.py ordner/                     # alle Bilder eines Ordners (auch Unterordner)
    python predict.py data/test                   # Unterordner = Klassennamen -> richtig/falsch wird angezeigt
    python predict.py --kamera                    # Live-Vorhersage im Kamerabild
    python predict.py --modell cnn --kamera       # bestimmtes Modell: cnn, cnn_augmentation, transfer, ...

Live-Kamera: LEERTASTE speichert das aktuelle Bild samt Vorhersage in out/kamera_vorhersagen/,
             Q / ESC beendet.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf

from config import IMG_SIZE, OUT_DIR
from training import MODEL_DIR, VARIANTS

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
UNSURE = 0.70      # darunter wird "unsicher" angezeigt


# --------------------------------------------------------------------------- Modell + Vorverarbeitung

def best_model_name():
    best, best_acc = None, -1.0
    for name in VARIANTS:
        f = OUT_DIR / name / "ergebnisse.json"
        if f.exists() and (MODEL_DIR / f"{name}.keras").exists():
            acc = json.loads(f.read_text(encoding="utf-8"))["accuracy"]["test"]
            if acc > best_acc:
                best, best_acc = name, acc
    return best


def load_model(name_or_path=None):
    if name_or_path is None:
        name_or_path = best_model_name()
        if name_or_path is None:
            raise SystemExit("Kein trainiertes Modell gefunden - zuerst train_cnn.py ausführen!")
    path = Path(name_or_path)
    if not path.suffix:
        path = MODEL_DIR / f"{name_or_path}.keras"
    if not path.exists():
        raise SystemExit(f"Modell {path} nicht gefunden.")
    classes_file = path.parent / "klassen.json"
    class_names = json.loads(classes_file.read_text(encoding="utf-8"))
    print(f"Modell: {path}")
    return keras.models.load_model(path), class_names


def prepare(rgb):
    """RGB-Bild (H x W x 3, uint8) -> Batch (1 x 128 x 128 x 3, float32 0..255) wie beim Training."""
    h, w = rgb.shape[:2]
    side = min(h, w)
    y, x = (h - side) // 2, (w - side) // 2
    square = rgb[y:y + side, x:x + side]
    resized = tf.image.resize(square, IMG_SIZE, method="bilinear")
    return tf.expand_dims(resized, 0)


def read_image(path):
    data = tf.io.read_file(str(path))
    return tf.io.decode_image(data, channels=3, expand_animations=False).numpy()


def predict(model, rgb):
    return model.predict_on_batch(prepare(rgb))[0]


# --------------------------------------------------------------------------- Bilddateien

def collect(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() in IMG_EXT:
            files.append(p)
    return files


def predict_files(model, class_names, paths, out_png):
    files = collect(paths)
    if not files:
        raise SystemExit("Keine Bilder gefunden.")
    rows = []
    for f in files:
        rgb = read_image(f)
        probs = predict(model, rgb)
        k = int(probs.argmax())
        truth = f.parent.name if f.parent.name in class_names else None
        mark = "" if truth is None else ("  richtig" if truth == class_names[k] else f"  FALSCH (ist: {truth})")
        unsure = "  (unsicher)" if probs[k] < UNSURE else ""
        print(f"{f.name:<45} -> {class_names[k]:<14} {probs[k]:6.1%}{unsure}{mark}")
        print("      " + "   ".join(f"{c}: {p:.1%}" for c, p in zip(class_names, probs)))
        rows.append((f, rgb, probs, truth))

    known = [r for r in rows if r[3] is not None]
    if known:
        correct = sum(class_names[int(r[2].argmax())] == r[3] for r in known)
        print(f"\nAccuracy auf {len(known)} Bildern mit bekannter Klasse: {correct / len(known):.1%}")
    save_grid(rows, class_names, out_png)


def save_grid(rows, class_names, path, max_images=40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = rows[:max_images]
    cols = min(8, len(rows))
    n_rows = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(2.1 * cols, 2.5 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (f, rgb, probs, truth) in zip(axes.flat, rows):
        k = int(probs.argmax())
        color = "black" if truth is None else ("green" if truth == class_names[k] else "red")
        ax.imshow(prepare(rgb)[0].numpy().astype("uint8"))
        ax.set_title(f"{class_names[k]} {probs[k]:.0%}\n{f.name[-28:]}", fontsize=7.5, color=color)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Übersicht gespeichert: {path}")


# --------------------------------------------------------------------------- Kamera

def live(model, class_names):
    import cv2
    from capture_images import GREEN, RED, WHITE, YELLOW, center_square, open_camera, put, shade

    cam = open_camera()
    if cam is None:
        return
    window = "Vorhersage (LEERTASTE = Bild speichern, Q = Ende)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1120, 630)
    save_dir = OUT_DIR / "kamera_vorhersagen"
    fps_t, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            crop, (cx, cy, side) = center_square(frame)
            probs = predict(model, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            k = int(probs.argmax())

            view = cv2.flip(frame, 1)
            h, w = view.shape[:2]
            cx = w - cx - side
            shade(view, 0, 0, cx, h, 0.65)
            shade(view, cx + side, 0, w, h, 0.65)
            color = GREEN if probs[k] >= UNSURE else YELLOW
            cv2.rectangle(view, (cx, cy), (cx + side - 1, cy + side - 1), color, 3)
            s = h / 720
            put(view, f"{class_names[k].upper()}  {probs[k]:.0%}", (20, int(60 * s)), 1.4 * s, color, 3)
            if probs[k] < UNSURE:
                put(view, "unsicher", (20, int(100 * s)), 0.8 * s, YELLOW, 2)
            for rank, i in enumerate(np.argsort(-probs)):   # Balken für alle Klassen
                y = int((150 + 40 * rank) * s)
                cv2.rectangle(view, (20, y - int(20 * s)), (20 + int(probs[i] * 260 * s), y + int(6 * s)),
                              GREEN if i == k else (120, 120, 120), -1)
                put(view, f"{class_names[i]} {probs[i]:.0%}", (25, y), 0.6 * s, WHITE, 1)
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-3)
            fps_t = now
            put(view, "Hohe Wahrscheinlichkeit heisst NICHT automatisch richtig!",
                (20, h - int(45 * s)), 0.55 * s, RED, 1)
            put(view, f"LEERTASTE Bild speichern   Q Ende   ({fps:.0f} FPS)", (20, h - int(15 * s)), 0.55 * s)
            cv2.imshow(window, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 32:
                save_dir.mkdir(parents=True, exist_ok=True)
                f = save_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{class_names[k]}_{probs[k]:.2f}.jpg"
                cv2.imwrite(str(f), crop)
                print(f"gespeichert: {f.name}   " + "   ".join(f"{c}: {p:.1%}" for c, p in zip(class_names, probs)))
    finally:
        cam.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Vorhersage mit gespeichertem CNN (Kapitel 8)")
    parser.add_argument("bilder", nargs="*", help="Bilddateien oder Ordner")
    parser.add_argument("--kamera", action="store_true", help="Live-Vorhersage im Kamerabild")
    parser.add_argument("--modell", help="cnn, cnn_augmentation, transfer, transfer_finetuning oder Pfad zu .keras")
    parser.add_argument("--ausgabe", default=str(OUT_DIR / "vorhersagen.png"), help="Übersichtsbild für Dateien")
    args = parser.parse_args()
    if not args.bilder and not args.kamera:
        parser.error("Bilder/Ordner angeben oder --kamera verwenden.")

    model, class_names = load_model(args.modell)
    print(f"Klassen: {class_names}\n")
    if args.bilder:
        predict_files(model, class_names, args.bilder, Path(args.ausgabe))
    if args.kamera:
        live(model, class_names)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
