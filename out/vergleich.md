# Übung 03 - Vergleich der Modelle

Automatisch erzeugt von `training.py` nach jedem Training.

| Variante | trainierbare Parameter | Epochen | Trainingszeit | Acc. train | Acc. validation | Acc. test | Test: serie_07 (gleiche Sitzung) | Test: spätere Testbilder (testserie_*) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Eigenes CNN | 105,891 | 41 | 13 s | 78.6% | 76.2% | 70.1% | 64.3% (42) | 75.6% (45) |
| Eigenes CNN + Data Augmentation | 105,891 | 120 | 84 s | 81.9% | 85.7% | 64.4% | 76.2% (42) | 53.3% (45) |
| Transfer Learning (MobileNetV2, Feature Extractor) | 3,843 | 30 | 23 s | 97.1% | 92.9% | 87.4% | 83.3% (42) | 91.1% (45) |
| Transfer Learning + Fine-Tuning | 1,514,563 | 60 | 63 s | 100.0% | 95.2% | 83.9% | 83.3% (42) | 84.4% (45) |

*Test: serie_07* = gemischte Serie aus derselben Aufnahmesitzung wie das Training; *spätere Testbilder* = an anderem Tag / Ort / mit anderem Exemplar aufgenommen (der eigentliche Generalisierungstest).

## Eigenes CNN

- Overfitting: Anzeichen für Overfitting: der Validierungs-Loss steigt nach Epoche 31 wieder an, während der Trainings-Loss weiter fällt.
- Lernkurven: [`cnn/lernkurven.png`](cnn/lernkurven.png)
- Konfusionsmatrix: [`cnn/konfusionsmatrix.png`](cnn/konfusionsmatrix.png)
- Beispiele: [`cnn/beispiele_richtig_falsch.png`](cnn/beispiele_richtig_falsch.png)
- Architektur: [`cnn/architektur.txt`](cnn/architektur.txt)

Sicherste Fehlklassifikationen im Test:

| Datei | ist | erkannt als | Wahrscheinlichkeit |
|---|---|---|---:|
| serie_07__handy_serie_07_004.jpg | handy | maus | 77% |
| testserie_01__handy_testserie_01_007.jpg | handy | maus | 77% |
| serie_07__handy_serie_07_003.jpg | handy | maus | 76% |
| testserie_01__handy_testserie_01_009.jpg | handy | maus | 74% |
| serie_07__handy_serie_07_005.jpg | handy | maus | 73% |
| testserie_01__flasche_testserie_01_014.jpg | flasche | handy | 71% |
| serie_07__handy_serie_07_002.jpg | handy | maus | 68% |
| testserie_01__handy_testserie_01_010.jpg | handy | maus | 68% |

## Eigenes CNN + Data Augmentation

- Overfitting: Anzeichen für Overfitting: der Validierungs-Loss steigt nach Epoche 112 wieder an, während der Trainings-Loss weiter fällt.
- Lernkurven: [`cnn_augmentation/lernkurven.png`](cnn_augmentation/lernkurven.png)
- Konfusionsmatrix: [`cnn_augmentation/konfusionsmatrix.png`](cnn_augmentation/konfusionsmatrix.png)
- Beispiele: [`cnn_augmentation/beispiele_richtig_falsch.png`](cnn_augmentation/beispiele_richtig_falsch.png)
- Architektur: [`cnn_augmentation/architektur.txt`](cnn_augmentation/architektur.txt)

Sicherste Fehlklassifikationen im Test:

| Datei | ist | erkannt als | Wahrscheinlichkeit |
|---|---|---|---:|
| testserie_01__flasche_testserie_01_013.jpg | flasche | maus | 99% |
| testserie_01__flasche_testserie_01_004.jpg | flasche | maus | 99% |
| testserie_01__flasche_testserie_01_015.jpg | flasche | maus | 99% |
| testserie_01__flasche_testserie_01_014.jpg | flasche | maus | 98% |
| testserie_01__flasche_testserie_01_002.jpg | flasche | maus | 97% |
| testserie_01__flasche_testserie_01_003.jpg | flasche | maus | 96% |
| testserie_01__handy_testserie_01_007.jpg | handy | maus | 87% |
| testserie_01__flasche_testserie_01_009.jpg | flasche | maus | 84% |

## Transfer Learning (MobileNetV2, Feature Extractor)

- Overfitting: Kaum Overfitting: Trainings- und Validierungskurven verlaufen nahe beieinander (Abstand 1% in Epoche 29).
- Lernkurven: [`transfer/lernkurven.png`](transfer/lernkurven.png)
- Konfusionsmatrix: [`transfer/konfusionsmatrix.png`](transfer/konfusionsmatrix.png)
- Beispiele: [`transfer/beispiele_richtig_falsch.png`](transfer/beispiele_richtig_falsch.png)
- Architektur: [`transfer/architektur.txt`](transfer/architektur.txt)

Sicherste Fehlklassifikationen im Test:

| Datei | ist | erkannt als | Wahrscheinlichkeit |
|---|---|---|---:|
| testserie_01__handy_testserie_01_008.jpg | handy | maus | 93% |
| serie_07__flasche_serie_07_001.jpg | flasche | maus | 87% |
| serie_07__flasche_serie_07_011.jpg | flasche | maus | 72% |
| serie_07__handy_serie_07_014.jpg | handy | maus | 62% |
| serie_07__handy_serie_07_006.jpg | handy | maus | 61% |
| testserie_01__handy_testserie_01_015.jpg | handy | flasche | 59% |
| serie_07__handy_serie_07_005.jpg | handy | maus | 59% |
| testserie_01__handy_testserie_01_011.jpg | handy | maus | 58% |

## Transfer Learning + Fine-Tuning

- Overfitting: Kaum Overfitting: Trainings- und Validierungskurven verlaufen nahe beieinander (Abstand 4% in Epoche 58).
- Lernkurven: [`transfer_finetuning/lernkurven.png`](transfer_finetuning/lernkurven.png)
- Konfusionsmatrix: [`transfer_finetuning/konfusionsmatrix.png`](transfer_finetuning/konfusionsmatrix.png)
- Beispiele: [`transfer_finetuning/beispiele_richtig_falsch.png`](transfer_finetuning/beispiele_richtig_falsch.png)
- Architektur: [`transfer_finetuning/architektur.txt`](transfer_finetuning/architektur.txt)

Sicherste Fehlklassifikationen im Test:

| Datei | ist | erkannt als | Wahrscheinlichkeit |
|---|---|---|---:|
| serie_07__flasche_serie_07_001.jpg | flasche | maus | 98% |
| testserie_01__handy_testserie_01_008.jpg | handy | maus | 96% |
| testserie_01__handy_testserie_01_011.jpg | handy | maus | 90% |
| serie_07__flasche_serie_07_011.jpg | flasche | maus | 83% |
| testserie_01__handy_testserie_01_009.jpg | handy | maus | 81% |
| testserie_01__handy_testserie_01_013.jpg | handy | maus | 77% |
| testserie_01__handy_testserie_01_001.jpg | handy | maus | 76% |
| testserie_01__handy_testserie_01_007.jpg | handy | maus | 75% |
