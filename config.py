"""
Gemeinsame Einstellungen für Übung 03 (Aufnahme, Aufteilung, Einlesen).
"""

from pathlib import Path

BASE = Path(__file__).resolve().parent

RAW_DIR = BASE / "rohbilder"     # hier speichert der Aufnahme-Assistent (pro Klasse und Serie)
DATA_DIR = BASE / "data"         # hier landen train/validation/test nach dem Aufteilen
OUT_DIR = BASE / "out"           # Plots, Übersichten
CLASSES_FILE = RAW_DIR / "klassen.json"

IMG_SIZE = (128, 128)            # einheitliche Größe beim Einlesen mit Keras (Kapitel 4)
BATCH_SIZE = 32
SEED = 42

SPLITS = ("train", "validation", "test")

# Präfixe der Serien-Ordner in rohbilder/<klasse>/
SERIES_PREFIX = "serie_"         # normale Aufnahmeserien -> werden auf train/validation/test verteilt
TEST_PREFIX = "testserie_"       # später aufgenommene Testbilder -> landen immer in test

# Geführte Aufnahmeserien (Kapitel 3). Jede Serie variiert bewusst etwas anderes.
# Serie für Serie werden ALLE Klassen nacheinander aufgenommen, damit jede Klasse
# dieselben Hintergründe/Lichtverhältnisse sieht (sonst lernt das CNN den Hintergrund).
SERIES = [
    {
        "titel": "Mitte, normaler Abstand",
        "anleitung": [
            "Gegenstand ungefaehr in die Bildmitte, ca. 40-60 cm Abstand.",
            "Drehe ihn waehrend der Aufnahme langsam und zeige alle Seiten.",
        ],
    },
    {
        "titel": "Nah an der Kamera",
        "anleitung": [
            "Gegenstand nah an die Kamera: er fuellt etwa das halbe Bild.",
            "Kippe und drehe ihn, zeige auch Ober- und Unterseite.",
        ],
    },
    {
        "titel": "Weit weg, ueberall im Bild",
        "anleitung": [
            "Gegenstand weit weg (klein im Bild).",
            "Wandere damit in alle Ecken und an die Raender des Bildes.",
        ],
    },
    {
        "titel": "Anderer Hintergrund",
        "anleitung": [
            "Wechsle den Hintergrund (anderer Tisch, Wand, Stoff, Papier...).",
            "Position, Abstand und Winkel weiter variieren.",
        ],
    },
    {
        "titel": "Andere Beleuchtung",
        "anleitung": [
            "Aendere das Licht: Lampe an/aus, naeher ans Fenster, Schatten.",
            "Schraege Blickwinkel, Gegenstand auch mal in der Hand halten.",
        ],
    },
    {
        "titel": "Gemischt A",
        "anleitung": [
            "Kombiniere alles: anderer Abstand, andere Lage, anderes Licht.",
            "Gegenstand liegen lassen, halten, schraeg stellen...",
        ],
        "split": "validation",
    },
    {
        "titel": "Gemischt B",
        "anleitung": [
            "Nochmal alles kombinieren - moeglichst anders als bisher.",
            "Andere Ecke im Raum, anderer Untergrund, andere Haltung.",
        ],
        "split": "test",
    },
]
# Serien 1-5 decken je eine Bedingung ab und gehen ins Training. Die gemischten Serien
# 6 und 7 sind für validation/test reserviert: so fehlt im Training keine Bedingung
# komplett, und validation/test enthalten trotzdem keine Nachbarbilder aus dem Training.
IMAGES_PER_SERIES = 14           # 7 Serien x 14 = 98 Bilder pro Klasse (Vorgabe: 50-100)

TEST_SERIES = {
    "titel": "Spaetere Testbilder",
    "anleitung": [
        "NEUE Bedingungen: anderer Raum / Tageszeit / Hintergrund.",
        "Wenn moeglich ein ANDERES Exemplar derselben Art verwenden!",
    ],
}
IMAGES_PER_TEST_SERIES = 15
