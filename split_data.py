"""
Übung 03 - Kapitel 4: Aufteilen der Daten in train / validation / test

Die Bilder in rohbilder/<klasse>/<serie>/ werden SERIENWEISE aufgeteilt:
Alle Bilder einer Aufnahmeserie landen gemeinsam im selben Bereich. Würde man
einzelne Bilder zufällig verteilen, lägen fast identische Nachbarbilder in
train UND test - die Test-Accuracy wäre dann viel zu optimistisch.

  - serie_01 .. serie_05  (je eine Bedingung: Mitte, nah, weit, Hintergrund, Licht) -> train
  - serie_06  (gemischt)                                                         -> validation
  - serie_07  (gemischt)                                                         -> test
  - testserie_*  (später unter neuen Bedingungen aufgenommen)                    -> test

Ergebnis:  data/train/<klasse>/, data/validation/<klasse>/, data/test/<klasse>/
           data/aufteilung.json (welche Serie wo gelandet ist)

Aufruf:
    python split_data.py
"""

import json
import shutil
import sys

from config import DATA_DIR, RAW_DIR, SERIES, SERIES_PREFIX, SPLITS, TEST_PREFIX

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def images_in(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def read_raw(raw_dir=RAW_DIR):
    """{klasse: {serienname: [bildpfade]}} für alle Klassen mit Bildern."""
    result = {}
    if not raw_dir.exists():
        return result
    # nur die im Assistenten festgelegten Klassen (alte, umbenannte Ordner ignorieren)
    classes_file = raw_dir / "klassen.json"
    wanted = set(json.loads(classes_file.read_text(encoding="utf-8"))) if classes_file.exists() else None
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        if wanted is not None and class_dir.name not in wanted:
            continue
        series = {}
        for s in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            imgs = images_in(s)
            if imgs:
                series[s.name] = imgs
        if series:
            result[class_dir.name] = series
    return result


def plan_split(series_names):
    """Ordnet die Serien einer Klasse train/validation/test zu.

    Die Zuordnung steht in config.SERIES ("split"); sie ist für alle Klassen gleich,
    damit keine Aufnahmebedingung nur bei einer Klasse im Training vorkommt.
    """
    plan = {split: [] for split in SPLITS}
    for name in sorted(series_names):
        if name.startswith(TEST_PREFIX):
            plan["test"].append(name)
        elif name.startswith(SERIES_PREFIX):
            nr = int(name[len(SERIES_PREFIX):])
            role = SERIES[nr - 1].get("split", "train") if nr <= len(SERIES) else "train"
            plan[role].append(name)
    return plan


def split(raw_dir=RAW_DIR, data_dir=DATA_DIR, verbose=True):
    raw = read_raw(raw_dir)
    if len(raw) < 2:
        raise SystemExit(f"Zu wenige Klassen mit Bildern in {raw_dir} gefunden "
                         f"({len(raw)}). Zuerst mit capture_images.py aufnehmen!")

    # nur die von uns erzeugten Unterordner löschen, nicht den ganzen data-Ordner
    for split_name in SPLITS:
        target = data_dir / split_name
        if target.exists():
            shutil.rmtree(target)

    info = {"klassen": {}}
    counts = {}
    warnings = []
    for klasse, series in raw.items():
        plan = plan_split(series.keys())
        info["klassen"][klasse] = plan
        counts[klasse] = {}
        for split_name, names in plan.items():
            target = data_dir / split_name / klasse
            target.mkdir(parents=True, exist_ok=True)
            n = 0
            for name in names:
                for img in series[name]:
                    # Serienname im Dateinamen behalten -> man sieht später, woher ein Bild stammt
                    shutil.copy2(img, target / f"{name}__{img.name}")
                    n += 1
            counts[klasse][split_name] = n
        if not plan["validation"] or not plan["test"]:
            warnings.append(f"Klasse '{klasse}': zu wenige Serien "
                            f"({len(series)}) - validation oder test bleibt leer!")

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "aufteilung.json").write_text(
        json.dumps({**info, "anzahl_bilder": counts}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    if verbose:
        print_table(info, counts)
        for w in warnings:
            print("  WARNUNG:", w)
    return info, counts


def print_table(info, counts):
    print()
    print(f"{'Klasse':<16}{'train':>8}{'validation':>12}{'test':>8}")
    print("-" * 44)
    total = {s: 0 for s in SPLITS}
    for klasse, c in counts.items():
        print(f"{klasse:<16}{c['train']:>8}{c['validation']:>12}{c['test']:>8}")
        for s in SPLITS:
            total[s] += c[s]
    print("-" * 44)
    print(f"{'Summe':<16}{total['train']:>8}{total['validation']:>12}{total['test']:>8}")
    print()
    print("Zuordnung der Serien:")
    for klasse, plan in info["klassen"].items():
        print(f"  {klasse}:")
        for s in SPLITS:
            print(f"    {s:<11} {', '.join(plan[s]) or '-'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    split()
    print(f"\nFertig. Daten liegen in: {DATA_DIR}")
