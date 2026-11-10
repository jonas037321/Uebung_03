"""
Übung 03: Assistent für die gesamte Übung (Aufnahme, Aufteilen, Training, Vorhersage)

Nach dem Start führt ein Assistent Schritt für Schritt durch die Übung:

  1  Klassen (Gegenstände) festlegen
  2  Trainingsbilder aufnehmen - geführt in 7 Serien mit unterschiedlichen
     Bedingungen (Position, Abstand, Winkel, Hintergrund, Licht, gemischt)
  3  Spätere Testbilder unter neuen Bedingungen aufnehmen
  4  Übersicht über alle aufgenommenen Bilder
  5  Daten serienweise in train / validation / test aufteilen  (split_data.py)
  6  Daten mit Keras einlesen und prüfen                        (load_data.py)
  7  Eigenes CNN trainieren                                     (train_cnn.py)
  8  Eigenes CNN mit Data Augmentation trainieren               (train_cnn.py --augmentation)
  9  Transfer Learning mit MobileNetV2                          (transfer_learning.py)
  10 Live-Vorhersage mit der Kamera                             (predict.py --kamera)

Die Bilder werden als quadratischer Ausschnitt aus der Bildmitte gespeichert:
    rohbilder/<klasse>/serie_01/<klasse>_serie_01_001.jpg
    rohbilder/<klasse>/testserie_01/...

Aufruf:
    python capture_images.py
"""

import json
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np

from config import (BASE, CLASSES_FILE, DATA_DIR, IMAGES_PER_SERIES, IMAGES_PER_TEST_SERIES,
                    OUT_DIR, RAW_DIR, SERIES, SERIES_PREFIX, TEST_PREFIX, TEST_SERIES)

WINDOW = "Aufnahme-Assistent (Fenster anklicken, dann Tasten druecken)"
AUTO_INTERVAL = 0.6      # Sekunden zwischen zwei automatischen Aufnahmen
COUNTDOWN = 3            # Sekunden Vorlauf, bevor die automatische Aufnahme startet
DUP_THRESHOLD = 4.0      # mittlere Pixeldifferenz (0..255), darunter gilt ein Bild als "fast gleich"
SAVE_SIZE = 480          # gespeicherte Bilder: 480 x 480 Pixel (Keras skaliert später auf 128 x 128)
MIN_CLASSES = 3

# Farben (BGR)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 255)
GREEN = (80, 220, 80)
RED = (60, 60, 255)
GREY = (170, 170, 170)


# --------------------------------------------------------------------------- Konsole

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def headline(text):
    print()
    print("=" * 72)
    print(f"  {text}")
    print("=" * 72)


def wait_enter(text="ENTER drücken, um fortzufahren ..."):
    input(f"\n  >> {text}")


def ask_yes(question, default=True):
    hint = "[J/n]" if default else "[j/N]"
    answer = input(f"  {question} {hint} ").strip().lower()
    if not answer:
        return default
    return answer in ("j", "ja", "y", "yes")


# --------------------------------------------------------------------------- Daten auf der Platte

def load_classes():
    if CLASSES_FILE.exists():
        return json.loads(CLASSES_FILE.read_text(encoding="utf-8"))
    return []


def save_classes(classes):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLASSES_FILE.write_text(json.dumps(classes, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_name(name):
    """Klassennamen als Ordnernamen tauglich machen: klein, keine Umlaute/Leerzeichen."""
    name = name.strip().lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss"), (" ", "_"), ("-", "_")):
        name = name.replace(a, b)
    return re.sub(r"[^a-z0-9_]", "", name)


def series_name(prefix, nr):
    return f"{prefix}{nr:02d}"


def count_images(folder):
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def test_series_numbers(classes):
    numbers = set()
    for klasse in classes:
        folder = RAW_DIR / klasse
        if folder.exists():
            for p in folder.iterdir():
                if p.is_dir() and p.name.startswith(TEST_PREFIX):
                    numbers.add(int(p.name[len(TEST_PREFIX):]))
    return sorted(numbers)


def training_missing(classes):
    """Liste der (serien_index, klasse) die noch nicht vollständig sind."""
    return [(i, k) for i in range(len(SERIES)) for k in classes
            if count_images(RAW_DIR / k / series_name(SERIES_PREFIX, i + 1)) < IMAGES_PER_SERIES]


def newest_raw_image_time():
    newest = 0.0
    if RAW_DIR.exists():
        for p in RAW_DIR.rglob("*.jpg"):
            newest = max(newest, p.stat().st_mtime)
    return newest


# --------------------------------------------------------------------------- Kamera + Zeichnen

def open_camera():
    print("  Kamera wird geöffnet ...")
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for index in range(3):
        for backend in backends:
            cam = cv2.VideoCapture(index, backend)
            if cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ok, frame = cam.read()
                if ok and frame is not None:
                    print(f"  Kamera {index} bereit ({frame.shape[1]}x{frame.shape[0]}).")
                    return cam
            cam.release()
    print("\n  FEHLER: Keine Kamera gefunden!")
    print("  - Ist die Kamera angeschlossen und nicht von Teams/Zoom/... belegt?")
    print("  - Windows: Einstellungen > Datenschutz > Kamera > Desktop-Apps erlauben")
    return None


def center_square(frame):
    h, w = frame.shape[:2]
    side = min(h, w)
    y, x = (h - side) // 2, (w - side) // 2
    return frame[y:y + side, x:x + side], (x, y, side)


def ascii_text(text):
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("Ä", "Ae"), ("Ö", "Oe"), ("Ü", "Ue"), ("ß", "ss")):
        text = text.replace(a, b)
    return text


def put(img, text, org, scale=0.7, color=WHITE, thick=2):
    """Text mit schwarzem Rand (gut lesbar auf jedem Hintergrund)."""
    text = ascii_text(text)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def shade(img, x1, y1, x2, y2, alpha=0.6):
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = (roi * (1 - alpha)).astype(np.uint8)


def thumb(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)


# --------------------------------------------------------------------------- Eine Serie aufnehmen

class Recorder:
    def __init__(self, cam):
        self.cam = cam
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1120, 630)

    def window_closed(self):
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1

    def read(self):
        ok, frame = self.cam.read()
        if not ok or frame is None:
            raise RuntimeError("Kamera liefert kein Bild mehr.")
        return frame

    def record(self, klasse, folder, target, kopf, anleitung):
        """Nimmt Bilder für eine Klasse/Serie auf.

        Rückgabe: "fertig", "weiter" (Serie übersprungen) oder "menue" (abgebrochen).
        """
        folder.mkdir(parents=True, exist_ok=True)
        existing = sorted(folder.glob("*.jpg"))
        next_nr = max((int(p.stem.rsplit("_", 1)[-1]) for p in existing), default=0) + 1
        count = len(existing)
        saved_now = []          # in dieser Sitzung gespeicherte Dateien (für "Rückgängig")
        last_thumb = None
        mode = "pause"          # pause | countdown | auto
        mode_since = 0.0
        last_shot = 0.0
        flash_until = 0.0
        hint, hint_until = "", 0.0

        def shoot(crop, check_duplicate):
            nonlocal next_nr, count, last_thumb, flash_until, hint, hint_until
            t = thumb(crop)
            if check_duplicate and last_thumb is not None and np.abs(t - last_thumb).mean() < DUP_THRESHOLD:
                hint, hint_until = "Fast gleiches Bild uebersprungen - Gegenstand bewegen!", time.time() + 1.2
                return
            path = folder / f"{klasse}_{folder.name}_{next_nr:03d}.jpg"
            cv2.imwrite(str(path), cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_now.append(path)
            next_nr += 1
            count += 1
            last_thumb = t
            flash_until = time.time() + 0.12

        while True:
            frame = self.read()
            crop, (cx, cy, side) = center_square(frame)
            now = time.time()

            if mode == "countdown" and now - mode_since >= COUNTDOWN:
                mode, last_shot = "auto", 0.0
            if mode == "auto" and now - last_shot >= AUTO_INTERVAL:
                last_shot = now
                shoot(crop, check_duplicate=True)

            if count >= target:
                self.show_message(frame, [f"Serie fertig: {count} Bilder von '{klasse}'", "Super!"], GREEN, 1.3)
                return "fertig"

            self.draw(frame, klasse, kopf, anleitung, count, target, mode, mode_since, now,
                      (cx, cy, side), flash_until > now, hint if hint_until > now else "")

            key = cv2.waitKey(1) & 0xFF
            if self.window_closed() or key in (27, ord("q"), ord("Q")):
                return "menue"
            if key == 32:  # Leertaste: Auto-Aufnahme starten / pausieren
                if mode == "pause":
                    mode, mode_since = "countdown", now
                else:
                    mode = "pause"
            elif key in (ord("s"), ord("S")):
                shoot(crop, check_duplicate=False)
            elif key in (ord("z"), ord("Z"), 8):
                if saved_now:
                    saved_now.pop().unlink(missing_ok=True)
                    count -= 1
                    hint, hint_until = "Letztes Bild geloescht", now + 1.0
                else:
                    hint, hint_until = "Nichts zum Rueckgaengigmachen", now + 1.0
            elif key in (ord("n"), ord("N")):
                return "weiter"

    def draw(self, frame, klasse, kopf, anleitung, count, target, mode, mode_since, now, box, flash, hint):
        cx, cy, side = box
        view = cv2.flip(frame, 1)                 # Spiegelbild wirkt natürlicher
        h, w = view.shape[:2]
        cx = w - cx - side                       # Rahmen ebenfalls spiegeln

        # Bereich außerhalb des gespeicherten Quadrats abdunkeln
        shade(view, 0, 0, cx, h, 0.65)
        shade(view, cx + side, 0, w, h, 0.65)
        frame_color = WHITE if flash else (GREEN if mode == "auto" else YELLOW)
        cv2.rectangle(view, (cx, cy), (cx + side - 1, cy + side - 1), frame_color, 8 if flash else 2)

        s = h / 720                               # Schrift an Auflösung anpassen
        shade(view, 0, 0, w, int(118 * s), 0.65)
        put(view, kopf, (15, int(32 * s)), 0.75 * s, GREY, 2)
        put(view, f"Klasse: {klasse.upper()}", (15, int(72 * s)), 1.15 * s, YELLOW, 3)
        put(view, f"{count}/{target} Bilder", (w - int(260 * s), int(72 * s)), 1.0 * s, WHITE, 2)
        bar_w = int((w - 30) * min(count / target, 1))
        cv2.rectangle(view, (15, int(92 * s)), (w - 15, int(106 * s)), (80, 80, 80), -1)
        cv2.rectangle(view, (15, int(92 * s)), (15 + bar_w, int(106 * s)), GREEN, -1)

        if mode == "pause":
            y0 = int(170 * s)
            shade(view, cx + 10, y0 - int(40 * s), cx + side - 10, y0 + int(150 * s), 0.6)
            put(view, f"Halte jetzt: {klasse.upper()}", (cx + 25, y0), 0.9 * s, YELLOW, 2)
            for i, line in enumerate(anleitung):
                put(view, line, (cx + 25, y0 + int((40 + 32 * i) * s)), 0.5 * s, WHITE, 1)
            put(view, "LEERTASTE = automatische Aufnahme starten", (cx + 25, y0 + int(130 * s)), 0.55 * s, GREEN, 2)
        elif mode == "countdown":
            left = COUNTDOWN - int(now - mode_since)
            put(view, str(left), (cx + side // 2 - int(30 * s), cy + side // 2 + int(30 * s)), 3 * s, YELLOW, 6)
            put(view, "Gleich geht's los - Gegenstand bewegen!", (cx + 25, cy + side - int(60 * s)), 0.65 * s)
        else:
            put(view, "AUFNAHME LAEUFT - langsam bewegen, drehen, verschieben!",
                (cx + 15, cy + side - int(60 * s)), 0.55 * s, GREEN, 2)

        if hint:
            put(view, hint, (cx + 15, cy + side - int(100 * s)), 0.6 * s, RED, 2)

        shade(view, 0, h - int(40 * s), w, h, 0.7)
        put(view, "LEERTASTE Auto start/pause   S Einzelbild   Z Rueckgaengig   N Serie ueberspringen   Q Menue",
            (15, h - int(13 * s)), 0.52 * s, WHITE, 1)
        cv2.imshow(WINDOW, view)

    def show_message(self, frame, lines, color, seconds):
        end = time.time() + seconds
        while time.time() < end:
            view = cv2.flip(self.read(), 1)
            h, w = view.shape[:2]
            s = h / 720
            shade(view, 0, 0, w, h, 0.6)
            for i, line in enumerate(lines):
                put(view, line, (40, int((h / 2 - 30 + 50 * i * s))), 1.0 * s, color if i == 0 else WHITE, 2)
            cv2.imshow(WINDOW, view)
            if (cv2.waitKey(1) & 0xFF) == 27 or self.window_closed():
                return

    def close(self):
        self.cam.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


# --------------------------------------------------------------------------- Menüpunkte

def explain_intro():
    clear()
    headline("Übung 03 – Aufnahme-Assistent für den eigenen CNN-Datensatz")
    print("""
  Ziel: Ein eigenes CNN soll mindestens 3 Gegenstände unterscheiden
  (z.B. Maus, Smartphone, Stift) – und zwar auch auf WIRKLICH NEUEN Bildern.

  Dafür brauchst du pro Klasse 50–100 Bilder, die sich bewusst unterscheiden:
    • Position im Bild       • Abstand (Größe)      • Blickwinkel/Rotation
    • Hintergrund            • Beleuchtung

  100 fast gleiche Bilder bringen kaum etwas: Das Netz lernt dann nur dieses
  eine Foto auswendig. Deshalb führt dich der Assistent durch {s} SERIEN mit je
  {n} Bildern pro Klasse – jede Serie ändert gezielt eine Bedingung.

  So läuft's ab:
    1  Klassen festlegen
    2  Trainingsbilder aufnehmen (Serie für Serie, alle Klassen)
    3  SPÄTER (anderer Tag/Raum, wenn möglich anderer Gegenstand derselben
       Art) eigene Testbilder aufnehmen
    5  Bilder serienweise in train / validation / test aufteilen
    6  Mit Keras einlesen und prüfen
    7–9  CNN trainieren, mit Augmentation, mit Transfer Learning
    10 Live-Vorhersage mit der Kamera

  Im Kamerafenster wird nur der helle QUADRATISCHE Ausschnitt gespeichert.
  Wichtig: Für Tastendruck zuerst ins Kamerafenster klicken!""".format(n=IMAGES_PER_SERIES, s=len(SERIES)))
    wait_enter()


def menu_classes():
    clear()
    headline("1  Klassen festlegen")
    classes = load_classes()
    if classes:
        print(f"\n  Aktuelle Klassen: {', '.join(classes)}")
    print(f"""
  Wähle mindestens {MIN_CLASSES} Gegenstände, die
    • du leicht zur Hand hast (am besten auch ein 2. Exemplar für die Tests),
    • sich deutlich voneinander unterscheiden.
  Beispiele: maus, smartphone, stift   |   tasse, kopfhoerer, fernbedienung

  Namen mit Komma getrennt eingeben (leer lassen = nichts ändern).""")
    raw = input("\n  Klassen: ").strip()
    if not raw:
        return
    new = []
    for part in raw.split(","):
        name = clean_name(part)
        if name and name not in new:
            new.append(name)
    if len(new) < MIN_CLASSES:
        print(f"\n  Es braucht mindestens {MIN_CLASSES} Klassen – nichts geändert.")
        wait_enter()
        return
    removed = [k for k in classes if k not in new and (RAW_DIR / k).exists()]
    save_classes(new)
    print(f"\n  Gespeichert: {', '.join(new)}")
    for k in removed:
        print(f"  Hinweis: Ordner {RAW_DIR / k} bleibt erhalten, wird aber nicht mehr verwendet.")
    wait_enter()


def run_capture(jobs, explain):
    """jobs: Liste von (klasse, ordner, ziel, kopfzeile, anleitung, neue_serie_text)."""
    cam = open_camera()
    if cam is None:
        wait_enter()
        return
    rec = Recorder(cam)
    try:
        for klasse, folder, target, kopf, anleitung, intro in jobs:
            if intro:
                print(intro)
            print(f"  -> {klasse}: {kopf}")
            result = rec.record(klasse, folder, target, kopf, anleitung)
            print(f"     {count_images(folder)}/{target} Bilder ({result})")
            if result == "menue":
                print("\n  Aufnahme unterbrochen – du kannst später genau hier weitermachen.")
                break
        else:
            print("\n  " + explain)
    finally:
        rec.close()
    wait_enter("ENTER drücken, um zum Menü zurückzukehren ...")


def menu_training():
    classes = load_classes()
    if len(classes) < MIN_CLASSES:
        print("\n  Zuerst Klassen festlegen (Menüpunkt 1)!")
        wait_enter()
        return
    missing = training_missing(classes)
    clear()
    headline("2  Trainingsbilder aufnehmen")
    if not missing:
        print("\n  Alle Trainingsserien sind bereits vollständig!")
        print("  Einzelne schlechte Bilder kannst du im Ordner rohbilder/ löschen –")
        print("  danach füllt dieser Menüpunkt die Lücken wieder auf.")
        wait_enter()
        return
    print(f"""
  Es fehlen noch {len(missing)} von {len(SERIES) * len(classes)} Serien (Klasse × Serie).

  Ablauf im Kamerafenster:
    1. Oben steht, welchen Gegenstand du jetzt zeigen sollst + was zu variieren ist.
    2. Gegenstand vorbereiten, ins Fenster klicken, LEERTASTE drücken.
    3. Nach 3 Sekunden Countdown macht das Programm alle {AUTO_INTERVAL} s ein Bild.
       BEWEGE den Gegenstand dabei ständig (drehen, verschieben, näher/weiter).
       Bewegst du nichts, werden fast gleiche Bilder automatisch übersprungen.
    4. Nach {IMAGES_PER_SERIES} Bildern geht es automatisch mit der nächsten Klasse weiter.

  Tasten: LEERTASTE Start/Pause · S Einzelbild · Z letztes Bild löschen
          N Serie überspringen · Q zurück ins Menü (Fortschritt bleibt gespeichert)

  Warum Serie für Serie für ALLE Klassen? Jede Klasse sieht so dieselben
  Hintergründe und dasselbe Licht. Sonst könnte das CNN einfach den Hintergrund
  lernen statt den Gegenstand ("Dataset Bias").""")
    wait_enter("ENTER drücken, um die Kamera zu starten ...")

    jobs = []
    last_series = None
    for i, klasse in missing:
        serie = SERIES[i]
        intro = None
        if i != last_series:
            lines = "\n".join(f"     {line}" for line in serie["anleitung"])
            intro = (f"\n  ---- Serie {i + 1}/{len(SERIES)}: {serie['titel']} ----\n{lines}\n"
                     f"     (Bei Hintergrund/Licht: jetzt umbauen, gilt für alle Klassen dieser Serie.)")
            last_series = i
        folder = RAW_DIR / klasse / series_name(SERIES_PREFIX, i + 1)
        kopf = f"Training - Serie {i + 1}/{len(SERIES)}: {serie['titel']}"
        jobs.append((klasse, folder, IMAGES_PER_SERIES, kopf, serie["anleitung"], intro))
    run_capture(jobs, "Alle Trainingsserien sind fertig! Nächster Schritt: Testbilder (3) – "
                      "am besten später unter anderen Bedingungen.")


def menu_test():
    classes = load_classes()
    if len(classes) < MIN_CLASSES:
        print("\n  Zuerst Klassen festlegen (Menüpunkt 1)!")
        wait_enter()
        return
    numbers = test_series_numbers(classes)
    nr = numbers[-1] if numbers else 1
    if numbers and all(count_images(RAW_DIR / k / series_name(TEST_PREFIX, nr)) >= IMAGES_PER_TEST_SERIES
                       for k in classes):
        nr += 1
    clear()
    headline(f"3  Spätere Testbilder aufnehmen (Testserie {nr})")
    print(f"""
  Diese Bilder kommen NUR in den Testdatensatz. Mit ihnen prüfst du, ob das CNN
  wirklich die KLASSE gelernt hat – oder nur deinen Gegenstand/Hintergrund.

  Darum laut Angabe: NICHT direkt im Anschluss an die Trainingsbilder aufnehmen!
    • anderer Tag, anderer Raum oder andere Tageszeit (anderes Licht)
    • ein neuer Hintergrund, der in den Trainingsserien nicht vorkam
    • wenn möglich ein ANDERES Exemplar (andere Maus, andere Trinkflasche ...)

  Pro Klasse {IMAGES_PER_TEST_SERIES} Bilder. Bedienung wie bei den Trainingsbildern.""")
    if not numbers and training_missing(classes):
        print("\n  Hinweis: Die Trainingsserien sind noch nicht vollständig.")
    if not ask_yes("\nJetzt Testbilder aufnehmen?"):
        return

    jobs = []
    for klasse in classes:
        folder = RAW_DIR / klasse / series_name(TEST_PREFIX, nr)
        if count_images(folder) < IMAGES_PER_TEST_SERIES:
            kopf = f"TEST - Testserie {nr}: neue Bedingungen"
            jobs.append((klasse, folder, IMAGES_PER_TEST_SERIES, kopf, TEST_SERIES["anleitung"], None))
    run_capture(jobs, "Testserie fertig! Nächster Schritt: Daten aufteilen (5).")


def menu_overview(pause=True):
    clear()
    headline("4  Übersicht der aufgenommenen Bilder")
    classes = load_classes()
    if not classes:
        print("\n  Noch keine Klassen festgelegt.")
        wait_enter()
        return
    tests = test_series_numbers(classes)
    cols = [series_name(SERIES_PREFIX, i + 1) for i in range(len(SERIES))]
    cols += [series_name(TEST_PREFIX, n) for n in tests]
    short = [c.replace(TEST_PREFIX, "T").replace(SERIES_PREFIX, "S") for c in cols]
    print("\n  " + f"{'Klasse':<16}" + "".join(f"{s:>5}" for s in short) + f"{'Serien':>8}{'Später':>8}")
    print("  " + "-" * (16 + 5 * len(cols) + 16))
    for klasse in classes:
        counts = [count_images(RAW_DIR / klasse / c) for c in cols]
        n_train = sum(counts[:len(SERIES)])
        n_test = sum(counts[len(SERIES):])
        ok = "✓" if n_train >= 50 else " "
        print("  " + f"{klasse:<16}" + "".join(f"{c:>5}" for c in counts) + f"{n_train:>7}{ok}{n_test:>8}")
    print(f"\n  S01..S{len(SERIES):02d} = Aufnahmeserien (Ziel je {IMAGES_PER_SERIES}; S06 -> validation, S07 -> test), "
          f"T01.. = spätere Testserien (Ziel je {IMAGES_PER_TEST_SERIES})")
    print("  ✓ = mindestens 50 Bilder (Vorgabe: 50–100 pro Klasse)")
    print(f"  Ordner: {RAW_DIR}")
    if pause:
        wait_enter()


def menu_split():
    clear()
    headline("5  Daten aufteilen in train / validation / test")
    print("""
  Die Bilder werden SERIENWEISE verteilt: Alle Bilder einer Aufnahmeserie
  landen gemeinsam in train, validation ODER test. Zufällig einzeln verteilt
  würden fast identische Nachbarbilder in train und test landen – die
  Test-Accuracy wäre dann viel zu gut und sagt nichts über neue Bilder aus.

    • Serien 1–5 (je eine Bedingung)  -> train       (ca. 70 %)
    • Serie 6 (gemischt A)            -> validation  (ca. 15 %)
    • Serie 7 (gemischt B)            -> test        (ca. 15 %)
    • spätere Testserien (Menü 3)     -> zusätzlich test

  Der Ordner data/ (train, validation, test) wird dabei neu erzeugt.""")
    if not ask_yes("\nJetzt aufteilen?"):
        return
    import split_data
    try:
        split_data.split()
        print(f"\n  Fertig! Ergebnis in {DATA_DIR} (Details: data/aufteilung.json)")
    except SystemExit as e:   # z.B. noch keine Bilder vorhanden
        print(f"\n  {e}")
    wait_enter()


def menu_load():
    clear()
    headline("6  Daten mit Keras einlesen und prüfen")
    print("\n  Keras wird geladen (dauert ein paar Sekunden) ...\n")
    import load_data
    try:
        load_data.main()
    except SystemExit as e:   # z.B. noch nicht aufgeteilt
        print(f"\n  {e}")
        wait_enter()
        return
    print("""
  Passt alles? Dann sind die Kapitel 1–4 erledigt. Weiter mit [7]: eigenes CNN trainieren.""")
    wait_enter()


def run_script(title, text, args):
    """Startet ein Trainings-/Vorhersageskript als eigenen Prozess (Ausgabe direkt in der Konsole)."""
    clear()
    headline(title)
    print(text)
    if not (DATA_DIR / "train").exists():
        print("\n  Zuerst Daten aufteilen (Menüpunkt 5)!")
        wait_enter()
        return
    if not ask_yes("\nJetzt starten?"):
        return
    print()
    result = subprocess.run([sys.executable, *args], cwd=BASE)
    if result.returncode == 0 and (OUT_DIR / "vergleich.md").exists() and args[0] != "predict.py":
        print(f"\n  Ergebnisse: {OUT_DIR}  (Übersicht aller Modelle: out/vergleich.md)")
    wait_enter()


def menu_cnn():
    run_script("7  Eigenes CNN trainieren (Kapitel 5)", """
  4 Conv2D-Schichten (16/32/64/128 Filter) mit MaxPooling, GlobalAveragePooling,
  Dense 64, Ausgabe mit Softmax. Training mit Early Stopping (val_loss).
  Ergebnis: out/cnn/ (Lernkurven, Konfusionsmatrix, richtige/falsche Beispiele).
  Dauer: je nach Rechner 1–5 Minuten.""", ["train_cnn.py"])


def menu_cnn_aug():
    run_script("8  Eigenes CNN mit Data Augmentation (Kapitel 7)", """
  Gleiches CNN, aber während des Trainings werden die Bilder zufällig gedreht,
  verschoben, gezoomt, gespiegelt und in Helligkeit/Kontrast verändert.
  Ergebnis: out/cnn_augmentation/ und der Vergleich in out/vergleich.md.
  Dauer: etwas länger als ohne Augmentation (mehr Epochen).""", ["train_cnn.py", "--augmentation"])


def menu_transfer():
    run_script("9  Transfer Learning mit MobileNetV2 (Kapitel 9)", """
  Vortrainiertes MobileNetV2 (ImageNet) als eingefrorener Feature Extractor,
  nur ein neuer Klassifikationskopf wird trainiert. Danach Fine-Tuning der
  obersten 30 Schichten mit kleiner Lernrate.
  Beim ersten Start werden die ImageNet-Gewichte (~9 MB) heruntergeladen.
  Ergebnis: out/transfer/, out/transfer_finetuning/ und out/vergleich.md.""",
               ["transfer_learning.py", "--finetuning"])


def menu_predict():
    run_script("10  Live-Vorhersage mit der Kamera (Kapitel 8)", """
  Lädt das Modell mit der besten Test-Accuracy und zeigt die erkannte Klasse
  samt Wahrscheinlichkeit im Kamerabild an.
  LEERTASTE speichert das Bild mit Vorhersage in out/kamera_vorhersagen/
  (gut für die Dokumentation in Kapitel 6), Q beendet.
  Probiere: neuer Hintergrund, anderes Licht, anderer Abstand, anderes Exemplar!""",
               ["predict.py", "--kamera"])


def recommendation(classes):
    if len(classes) < MIN_CLASSES:
        return "1", "Klassen festlegen"
    if training_missing(classes):
        return "2", "Trainingsbilder aufnehmen"
    if not test_series_numbers(classes):
        return "3", "Testbilder aufnehmen (am besten später / unter anderen Bedingungen)"
    info = DATA_DIR / "aufteilung.json"
    if not info.exists() or info.stat().st_mtime < newest_raw_image_time():
        return "5", "Daten aufteilen"
    if not (OUT_DIR / "beispielbilder.png").exists():
        return "6", "Daten mit Keras einlesen und prüfen"
    for nr, name, text in (("7", "cnn", "Eigenes CNN trainieren"),
                           ("8", "cnn_augmentation", "CNN mit Data Augmentation trainieren"),
                           ("9", "transfer", "Transfer Learning")):
        result = OUT_DIR / name / "ergebnisse.json"
        if not result.exists() or result.stat().st_mtime < info.stat().st_mtime:
            return nr, text
    return "10", "Live-Vorhersage mit der Kamera ausprobieren"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    explain_intro()
    actions = {"1": menu_classes, "2": menu_training, "3": menu_test,
               "4": menu_overview, "5": menu_split, "6": menu_load,
               "7": menu_cnn, "8": menu_cnn_aug, "9": menu_transfer, "10": menu_predict}
    while True:
        classes = load_classes()
        nr, text = recommendation(classes)
        clear()
        headline("Hauptmenü – Übung 03: CNN mit eigenen Bildern")
        if classes:
            print(f"\n  Klassen: {', '.join(classes)}")
        print(f"""
  [1]  Klassen festlegen / ändern
  [2]  Trainingsbilder aufnehmen          (Kapitel 3, geführt in {len(SERIES)} Serien)
  [3]  Spätere Testbilder aufnehmen       (Kapitel 3, "Wichtiger Test")
  [4]  Übersicht: was ist schon aufgenommen?
  [5]  Daten aufteilen: train/validation/test   (Kapitel 4)
  [6]  Daten mit Keras einlesen und prüfen      (Kapitel 4)
  [7]  Eigenes CNN trainieren                   (Kapitel 5)
  [8]  Eigenes CNN mit Data Augmentation        (Kapitel 7)
  [9]  Transfer Learning mit MobileNetV2        (Kapitel 9)
  [10] Live-Vorhersage mit der Kamera           (Kapitel 6 + 8)
  [0]  Beenden""")
        print(f"\n  Empfohlener nächster Schritt: [{nr}] {text}")
        choice = input("\n  Auswahl (ENTER = Empfehlung): ").strip() or nr
        if choice == "0":
            print("\n  Bis zum nächsten Mal!\n")
            return
        if choice in actions:
            actions[choice]()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Abgebrochen.\n")
