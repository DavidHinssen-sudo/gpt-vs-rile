# GPT vs. RILE: automatisierte Links-Rechts-Verortung von Parteiprogrammen

Code-Arbeitsprobe zur Bachelorarbeit *"How well does GPT reconstruct parties' left–right
positions compared to MARPOR's RILE?"* (Politikwissenschaft B.A., Universität Tübingen,
Erstprüfer Jakub Sowula, eingereicht 30.09.2025, Note 1,0).

---

## Fragestellung

Die RILE-Skala des Manifesto Project (MARPOR) ist einer der meistgenutzten Indikatoren für
Parteipositionen, und seit Jahren wegen Kodierfehlern und Inkonsistenzen in der Kritik.
Die Arbeit prüft, ob ein LLM (GPT) als reproduzierbarer Gegencheck taugt: Ich lasse GPT
dieselben Quasi-Sätze klassifizieren, die menschliche MARPOR-Coder kodiert haben, aggregiere
zu einem RILE-analogen Index und vergleiche beide gegen einen externen Expertenbenchmark.

**Daten:** 49 Wahlen in sieben Ländern (DE, ES, UK, DK, NO, NL, GR), späte 1990er bis 2020er.
Texte aus der Manifesto Project Database (Lehmann et al. 2024), Benchmark aus
Gemenis & Bruinsma (2020) auf Basis von CHES-Expertendaten.

**Ergebnis in einem Satz:** Gleichstand im Durchschnitt, aber bedingt: wo MARPORs
Humankodierung stark ist, gewinnt RILE; wo sie schwach ist, ist GPT genauer. Über wiederholte
Läufe erreicht GPT eine höhere Test-Retest-Konsistenz als die publizierte Intercoder-Reliabilität
menschlicher Coder. LLMs sind damit Ergänzung, nicht Ersatz: brauchbar, um verdächtige
Zeitpunkte für gezielte Audits zu markieren.

## Was hier drin ist

| Datei | Inhalt |
| --- | --- |
| `classify_manifestos.py` | Klassifikations-Pipeline: CSV mit Quasi-Sätzen rein, Labels (left/right/neutral) raus. Batching, JSON-Output, Fehlerbehandlung, Merge über Zeilen-IDs. |
| `rile_score.py` | Aggregation der Labels zum RILE-analogen Score pro Wahlprogramm. |
| `requirements.txt` | `pandas`, `openai`. |

```bash
export OPENAI_API_KEY=sk-...
python classify_manifestos.py --input data/SPDGER2013.csv --output results/
python rile_score.py results/*.csv
```

Die Kürzel folgen dem Schema `[Parteikürzel][ISO-Ländercode][Wahljahr]`, also
`SPDGER2013` für das SPD-Programm zur Bundestagswahl 2013.

## Designentscheidungen, die im Code sichtbar sind

**Drei Labels statt 56 Kategorien.** Der erste Versuch, MARPORs 56 Kategorien direkt auf
Links/Rechts/Neutral zu mappen, produzierte überlange Definitionen, und "Neutral" wurde zur
Restkategorie, in der alles Uneindeutige landete. Drei Korrekturen: Definitionen auf die
zentralen Unterscheidungen eingedampft; die Aufgabe von vornherein als Dreierwahl formuliert,
damit Neutral konzeptionell gleichwertig ist; umstrittene Kategorien wie 305 *Political Authority*
ausgeschlossen, die MARPOR als "rechts" führt und die in der Literatur als
Hauptfehlerquelle des RILE gelten (Mikhaylov et al. 2012).

**JSON statt Tabelle oder Fließtext.** Getestet wurden alle drei Ausgabeformate. JSON
parst zuverlässig und geht direkt in ein DataFrame. Jede Eingabezeile trägt ihre ID mit,
sodass Antworten auch dann korrekt zugeordnet werden, wenn das Modell die Reihenfolge ändert
oder Zeilen auslässt.

**Fehlende Labels bleiben leer.** Ein Batch mit kaputtem JSON kostet höchstens `batch_size`
Zeilen; die bleiben unklassifiziert und werden am Ende gezählt, statt still verloren zu gehen.
Bei einem Messinstrument ist ein sichtbares Loch besser als ein unsichtbares.

**Parameter:** `temperature=0.2` (Wang et al. 2024), Batchgröße variiert und ohne
substanziellen Effekt getestet, daher aus Effizienzgründen groß gewählt. Hauptmodell GPT-4o;
GPT-5 nur explorativ als Sensitivitätscheck.

## Transparenz zur Fassung

Der Klassifikations-Code ist der Hauptskript aus dem Anhang der Arbeit (S. 29-32).
Gegenüber der eingereichten Fassung ist er **strukturell aufgeräumt, nicht inhaltlich geändert**:
Pfade und Parameter laufen jetzt über die Kommandozeile statt über hartkodierte Variablen,
der API-Key kommt aus einer Umgebungsvariable statt aus einer Datei, Prompt-Aufbau und
Batch-Schleife stecken in benannten Funktionen, und Docstrings sind ergänzt. Prompt,
Label-Definitionen, Aggregationslogik und Ausgabeformat sind unverändert.

`rile_score.py` ist neu geschrieben und rekonstruiert die im Methodenteil (Abschnitt 4.7)
beschriebene Aggregation, die im Anhang nicht abgedruckt war.

## Referenzen

Gemenis, K. & Bruinsma, B. (2020): *Replication Data for: Challenging the Manifesto Project
Data Monopoly.* Harvard Dataverse. | Lehmann, P. et al. (2024): *Manifesto Project Dataset.*
| Mikhaylov, S., Laver, M. & Benoit, K. (2012): Coder Reliability and Misclassification in the
Human Coding of Party Manifestos. *Political Analysis* 20(1), 78-91.
