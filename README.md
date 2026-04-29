# The logo detector challenge

## Approccio alla challenge

Si è affrontata la challenge lavorando su due strade complementari:

### Generatore
- Cut and paste su background facendo trasformazioni geometriche affini/prospettiche per forzare invarianza rispetto a posizione, scala e diverse prospettive del logo.
- Trasformazioni di colore e luminosità per rendere il modello invariante rispetto a colore e differenze di illuminazione.
- In più, per cercare di forzare il modello a essere invariante rispetto agli artefatti generati dal cut and paste, si sono generate per ogni background e seed di trasformazioni diverse versioni con diverse tipologie di blending, in maniera da forzare invarianza rispetto a possibili artefatti.
- Uso di COCO 2017 val e DTD (Describable Textures Dataset) per rendere il modello invariante al background: il primo per immagini più realistiche, il secondo specifico per le possibili texture dietro al logo.
- Per evitare falsi positivi, un numero configurabile di esempi viene generato senza logo, così da rendere il modello meno sensibile ai falsi positivi.
- Aggiunte trasformazioni per compensare cali di risoluzione, rumore del sensore o movimento, insieme a Dropout per forzare un utilizzo più distribuito delle caratteristiche del logo piuttosto che l’eventuale dipendenza da una singola parte.

### Detector
- Caricamento pesi da DINOv2 patch14 (small/base) su un ViT: imparare da zero un feature extractor su un dataset sintetico è difficile, mentre accedere a pesi preaddestrati consente di convergere prima e con performance migliori. In particolare, si è scelto DINO perché le feature generate durante un training self-supervised sono più robuste e generalizzano meglio rispetto a un modello zero-shot, oltre a superare le performance di ImageNet. Inoltre, DINOv2 rispetto a DINOv1 ha patch embedding 14 invece che 16, quindi maggiore risoluzione.
- Il feature extractor è congelato (frozen).
- Per garantire sufficiente contesto spaziale è aggiunta una FPN, in particolare una versione semplificata di DPT.
- Per avere maggiore segnale rispetto a una regressione di due soli valori (x, y), si è scelto un approccio con una testa che produce una heatmap e una loss basata su CenterNet, cercando di predire una heatmap dove si prende il logit/probabilità di score più alto. Questo consente maggiore supervisione e quindi convergenza più rapida.

### Validazione
- Per il validation set si è stampato il logo e catturate più foto in diversi scenari indoor/outdoor, variando prospettiva e scala per misurare le performance del modello.
- Il piccolo dataset è stato labelizzato utilizzando LabelStudio e uno script custom per convertire le label in formato Underfolder. Il dataset è disponibile nella cartella `data`.

### Metriche
- Sono state implementate due metriche dedicate:
  - **LocalizationAccuracy**: conta quanti loghi hanno un errore < 10%.
  - **LogoPresenceAccuracy**: misura l’accuratezza sia quando il logo è presente sia quando non è presente, così da stimare falsi positivi e falsi negativi.

---

## Riferimenti (paper letti durante la challenge)
- *Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection*
- *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*
- *Deep Learning Logo Detection with Data Expansion by Synthesising Context*
- *On Pre-Trained Image Features and Synthetic Images for Deep Learning*

## Uso

Una volta installate le dipendenze:

```bash
uv sync  # oppure: pip install .
```

è possibile usare lo script `scripts/eval.py` per:
- valutare il modello su una cartella specifica
- oppure calcolare direttamente le metriche su un dataset di test definito nel config

## Nota

> Il progetto usa [`ezconfy`](https://github.com/alessioarcara/EzConfy), una **mia** libreria recentemente creata per gestire file di configurazione YAML. Qui viene usata per validare il config rispetto a `configs/schema.yaml` e per fare object instantiation direttamente da YAML tramite `_target_type_` e `_init_args_`.

## Struttura del repository

```text
├── checkpoints/        # Model checkpoints salvati
├── configs/            # Configurazioni YAML degli esperimenti
│   ├── base.yaml       # Configurazione principale
│   └── schema.yaml     # Schema della configurazione
├── data/               # Dataset e asset usati dal progetto
│   ├── raw/            # Dati esportati da LabelStudio
│   ├── training/       # Asset per la generazione sintetica
│   │   ├── backgrounds/ # Background usati dal generatore
│   │   └── logos/      # Loghi sorgente
│   └── validation/     # Validation set reale in formato Underfolder
├── notebooks/
│   └── visualize_generator.ipynb  # Notebook per visualizzare il generatore
├── scripts/            # Entry point e utility
│   ├── convert_labelstudio_data_to_underfolder.py  # Conversione LabelStudio
│   ├── download_backgrounds.sh  # Download dei background
│   ├── eval.py         # Script di evaluation
│   └── train.py        # Script di training
├── src/                # Codice principale del progetto
│   ├── data.py         # Dataset, dataloader
│   ├── generator/      # Pipeline di generazione cut-and-paste
│   ├── models/         # Architetture del detector
│   ├── training/       # Training loop, loss, metriche e callback
│   └── utils/          # Utility condivise
```
