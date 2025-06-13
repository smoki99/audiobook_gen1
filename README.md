# Audiobook Generator mit Fish Speech TTS

Dieses Projekt enthält Skripte zur automatisierten Generierung von Hörbüchern mithilfe der Fish Speech TTS API.

## Voraussetzungen

- Python 3.8 oder höher
- Fish Speech TTS API Server (lokal oder remote)
- Folgende Python-Pakete:
  - requests
  - ormsgpack
  - pydub (für Audio-Konkatenation)
  - librosa (für Geschwindigkeitsanpassung)
  - soundfile (für Audiobearbeitung)
  - fish_speech (inkl. utils)

## Installation

1. Klonen Sie dieses Repository:
```bash
git clone https://github.com/yourusername/audiobook-generator.git
cd audiobook-generator
```

2. Installieren Sie die benötigten Pakete:
```bash
pip install requests ormsgpack pydub librosa soundfile
# Stellen Sie sicher, dass fish_speech installiert ist
```

## Verwendung

### Verzeichnisstruktur

```
audiobook-generator/
├── audiobook_generator.py  # Hauptskript
├── fish_speech_tts.py      # TTS Client
├── input/                  # Eingabeverzeichnis für Textdateien
│   └── kapitel1.txt
│   └── kapitel2.txt
├── output/                 # Ausgabeverzeichnis für Audiodateien
├── voices/                 # Referenzstimmen
│   ├── calm.wav
│   ├── excited.wav
│   └── serious.wav
└── voice_config.json       # Konfigurationsdatei für Stimmen
```

### Textdateien-Format

Textdateien sollten im Eingabeverzeichnis (.txt) gespeichert werden und können Stimmtags im folgenden Format enthalten:

```
[stimmname]Der zu sprechende Text.
```

Beispiel:
```
[default]Dies ist ein Text mit der Standard-Stimme.
[calm]Hier spricht eine ruhige Stimme, die sehr entspannt klingt.
[excited]Wow! Das ist ja fantastisch! Ich kann es kaum erwarten!
```

### Stimmen-Konfiguration

Die `voice_config.json` enthält Konfigurationen für jede Stimme:

```json
{
  "default": {
    "reference_audio": null,
    "reference_text": null,
    "temperature": 0.8,
    "top_p": 0.8,
    "repetition_penalty": 1.1,
    "speed": 1.0
  },
  "calm": {
    "reference_audio": "voices/calm.wav",
    "reference_text": "Alles ist ruhig und friedlich, ganz entspannt und gelassen.",
    "temperature": 0.7,
    "speed": 0.9
  }
}
```

### Ausführung

Starten Sie den Generator mit:

```bash
python audiobook_generator.py input_dir output_dir voice_config.json
```

Beispiel:
```bash
python audiobook_generator.py input/ output/ voice_config.json
```

## Parameter für Stimmen

Sie können folgende Parameter für jede Stimme in der `voice_config.json` konfigurieren:

- `reference_audio`: Pfad zur Referenz-Audiodatei
- `reference_text`: Text der Referenz-Audio
- `temperature`: Sampling-Temperature (0.1-2.0, höher = mehr Variation)
- `top_p`: Top-p Sampling (0.1-1.0)
- `repetition_penalty`: Wiederholungsstrafe (1.0-2.0)
- `chunk_length`: Chunk-Länge für Synthese
- `max_new_tokens`: Maximale neue Tokens
- `output_format`: Ausgabeformat (wav, mp3, flac)
- `speed`: Geschwindigkeitsfaktor (0.5 = halbe Geschwindigkeit, 1.0 = normal, 2.0 = doppelte Geschwindigkeit)

## Beispiel-Ausgabe

Für die Eingabedatei `kapitel1.txt` mit 3 Zeilen würden folgende Dateien generiert:
- `output/kapitel1_000.wav`
- `output/kapitel1_001.wav`
- `output/kapitel1_002.wav`
- `output/kapitel1_concat.wav` (Zusammenführung aller Audiodateien)