# Segmentierendes Neuronales Netz

Dieses Readme zeigt, wie man das segmentierende neuronale Netz aufbaut.

## Erläuterung der Dateien und Verzeichnisse im Repo:
* Verzeichnis "chunks": 
Hier werden die elementaren Streckenelemente des Fahrbahngenerators gespeichert. 
Aus diesen wird im Carcassone-Prinzip die Strecke aufgebaut.

* Verzeichnis "data":
Hier sind die Trainings- und Validierungsdatensätze gespeichert.

* Verzeichnis "log":
Hier speichert Tensorflow den Trainingsverlauf und Fortschritt.

* Verzeichnis "overlays":
Hier sind die Störelemente gespeichert, die der Simulator zufällig auf die Fahrbahn abbildet.

* Verzeichnis "white_box":
Hier sind Bilder von Hindernissen gespeichert, die der Simulator zufällig auf die Fahrbahn abbildet. 
Damit wird die Klasse "Hindernisse" trainiert.

* Datei "config.py":
Diese Datei enthält alle wichtigen Kalibrierungen für den Fahrbahngenerator und das Training vom 
neuronalen Netz.

* Datei "curve.py":
In dieser Datei wird die elementare Kurve für den Fahrbahnsimulator gebaut und im Verzeichnis 
"chunks" gespeichert.

* Datei "dataset.py":
Diese Datei enthält nützliche Funktionen für das Erstellen und Bearbeiten von Tensorflow-Datensätzen.

* Datei "generic.py":
Diese Datei enthält die Elternklasse für die Klassen "curve", "intersection" und "line".

* Datei "imports.py":
Diese Datei gibt den Überblick über alle Importanweisungen der einzelnen Programme. 
In jeder Python-Datei wird ausschließlich diese Datei importiert.

* Datei "intersection.py":
In dieser Datei wird die elementare Kreuzung für den Fahrbahnsimulator gebaut und im Verzeichnis 
"chunks" gespeichert.

* Datei "line.py":
In dieser Datei wird die elementare Gerade für den Fahrbahnsimulator gebaut und im Verzeichnis 
"chunks" gespeichert.

* Datei "main.py":
In dieser Datei können die wichtigen Funktionen der Anwendungen ausgeführt werden.

* Datei "model.json":
Diese Datei speichert die Architektur des neuronalen Netzwerks.

* Dateien "model.py" und "model_deep.py":
Mit diesen Dateien kann die Architektur des neuronalen Netzes erstellt und in "model.json" abgespeichert werden.

* Datei "parameter.h5":
Dieser Datei speichert die trainierten Parameter des neuronalen Netzwerks.

* Datei "road.py":
Diese Datei erstellt aus den Streckenelementen eine Fahrbahn für den Fahrbahngenerator.

* Datei "test.py":
Diese Datei enthält wichtige Metriken, um die Qualität des neuronalen Netzes zu quantifizieren.

* Datei "train.py":
Diese Datei enthält alle Funktionen, um das neuronale Netzwerk zu trainieren.

* Datei "visualization.py":
Diese Datei erstellt aus den zusammengesetzten Bildern der Fahrbahn, einzelne Aufnahmen aus Sicht der Fahrzeugkamera.
Die Bilder erhalten gleichzeitig eine Etikette. Die Stichproben werden im Verzeichnis "data" gespeichert.

## Architektur des neuronalen Netzes
Der erste Schritt für den Aufbau eines neuronalen Netzes, ist die Architektur des Netzwerks aufzubauen.
Diese wird in der Datei "model.json" gespeichert. Vor der Benutzung des Netzwerks muss diese importiert werden.
Es bestehen zwei PY-Dateien um verschiedene Varianten des U-Nets zu erstellen.
Folgender Befehl baut das klassische U-Net:

``` bash
python model_deep.py
```

Für eine schlanke Version des U-Nets, führt ihr diesen Befehl aus:

``` bash
python model.py
```

## Synthetische Trainingsdaten generieren
Die synthetischen Daten des Generators werden in den Verzeichnissen "data/{images und annotations}/{trainset oder valset}/synthetic_set/" gespeichert.
Das Bild wird in dem Verzeichnis "images" und die gleichnamigen Etiketten im Verzeichnis "annotations" gespeichert.
Um etikettierte Stichproben zu generieren, kann folgender Befehl ausgeführt werden.

``` bash
python main.py --program generator
```

Der Simulator baut die virtuelle Umgebung nach dem Carcasonne-Prinzip auf. 
Solltet ihr die elementaren Bauteile des Simulators geändert haben, könnt ihr mit folgendem Befehl die Elemente neu generieren:

``` bash
python main.py --program create_elements
```

## Reale Trainingsdaten einrichten

Die realen Stichproben werden in den Verzeichnissen  "data/{images und annotations}/{trainset oder valset}/set_{index}/" gespeichert.
Dabei wird jede Bildsequenz in ein eigenes Verzeichnis gespeichert und erhält einen eigenen Index.
Wichtig ist, dass Bild und Etikette den selben Namen haben.

## Netzwerk trainieren

Das Netzwerk kann entweder vortrainierte Parameter verwenden oder von Neuem beginnen. 
Wenn vortrainierte Parameter benutzt werden, muss die Variable Config.pretrain den Wert True haben.
Das Verzeichnis mit den Trainingsdaten wird mit der Variable Config.train_path bestimmt.
Die Validierungsdaten sucht das Programm unter der Variable Config.val_path.
Wenn sich das Netzwerk verbessdert, speichert das Programm die neuen Parameter in die Datei "parameters.h5".

Das Netzwerk wird mit folgendem Befehl trainiert:

``` bash
python main.py --program train
```

Folgender Befehl startet Tensorboard, um den Trainingsverlauf zu visualisieren:
``` bash
python main.py --program debug
```

## Netzwerk testen

Folgender Befehl bestimmt die Genauigkeit der Lokalisierung der gesamten Anwendung.

``` bash
python main.py --program test
```