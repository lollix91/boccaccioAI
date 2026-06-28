# BoccaccioAI

**Modello linguistico nativo italiano da 1 miliardo di parametri, addestrato da zero.**

*De Lauretis Tech*

---

## Panoramica

BoccaccioAI e' un modello linguistico decoder-only da circa 1 miliardo di parametri, addestrato interamente da zero su testo italiano estratto dal dataset CulturaX. Il modello e' ottimizzato per compiti di generazione testuale e Question & Answering in lingua italiana.

Caratteristiche principali:

- Tokenizer BPE custom con vocabolario da 32.000 token, addestrato specificamente su testo italiano.
- Grouped Query Attention (GQA) con rapporto 4:1 per ridurre la memoria in fase di inferenza.
- Funzione di attivazione SwiGLU per una migliore efficienza di addestramento.
- Rotary Position Embeddings (RoPE) per la codifica posizionale relativa.
- Compatibilita' con Flash Attention 2 per l'accelerazione delle operazioni di attenzione.

---

## Architettura

| Componente         | Specifica                          |
|--------------------|------------------------------------|
| Parametri          | ~1B                                |
| Hidden Size        | 2048                               |
| Layer              | 24                                 |
| Attention Heads    | 16 (GQA 4:1, 4 KV heads)          |
| FFN                | SwiGLU (intermediate 5504)         |
| Posizionale        | RoPE (theta 10000)                 |
| Normalizzazione    | RMSNorm (eps 1e-5)                 |
| Contesto           | 2048 token                         |
| Vocabolario        | 32.000 token (BPE italiano)        |
| Embedding Condivisi| Input embedding = LM head          |
| Dropout            | 0.0 (standard pre-training LLM)    |

---

## Struttura del Progetto

```
boccaccioAI/
|-- configs/
|   |-- model.yaml              # Configurazione architettura modello
|   |-- tokenizer.yaml          # Configurazione tokenizer BPE
|   |-- training.yaml           # Iperparametri pre-training e fine-tuning
|-- scripts/
|   |-- 01_train_tokenizer.sh   # Fase 1: addestramento tokenizer
|   |-- 02_preprocess_data.sh   # Fase 2: pipeline dati (download, filtro, tokenizzazione)
|   |-- 02_5_smoke_test.sh      # Fase 2.5: smoke test su GPU locale (modello nano)
|   |-- 03_pretrain.sh          # Fase 3: pre-training modello 1B
|   |-- 04_finetune.sh          # Fase 4: instruction fine-tuning
|   |-- 05_evaluate.sh          # Fase 5: valutazione e test
|   |-- vm_setup.sh             # Setup VM Hetzner per Fasi 1-2
|   |-- run_fases_1_2.sh        # Esecuzione Fasi 1-2 su VM Hetzner
|-- src/
|   |-- data/
|   |   |-- download.py         # Download CulturaX italiano
|   |   |-- filter.py           # Filtraggio e deduplicazione documenti
|   |   |-- tokenize_corpus.py  # Pre-tokenizzazione in formato binario
|   |   |-- dataset.py          # Dataset PyTorch per il training
|   |-- inference/
|   |   |-- generate.py         # Generazione testo e Q&A
|   |-- model/
|   |   |-- config.py           # Dataclass configurazione modello
|   |   |-- attention.py        # Grouped Query Attention con RoPE
|   |   |-- layers.py           # SwiGLU FFN, RMSNorm, blocco transformer
|   |   |-- transformer.py      # Modello completo BoccaccioForCausalLM
|   |-- tokenizer/
|   |   |-- train_tokenizer.py  # Script addestramento tokenizer BPE
|   |-- training/
|       |-- lightning_module.py # Modulo PyTorch Lightning
|       |-- scheduler.py        # Cosine schedule con warmup
|       |-- callbacks.py        # Callback di training
|-- requirements.txt            # Dipendenze complete (training GPU)
|-- requirements-vm.txt         # Dipendenze minime (Fasi 1-2, VM senza GPU)
|-- README.md
|-- AGENTS.md
```

---

## Requisiti

- Python 3.10+
- CUDA 12.x con GPU compatibile (H100 consigliata per il training completo)
- PyTorch 2.3+

Il progetto utilizza due file di dipendenze separati in base all'ambiente di esecuzione:

### `requirements.txt` - Ambiente completo (training GPU)

Include tutte le dipendenze necessarie per il pre-training e fine-tuning del modello su GPU. Richiede una GPU CUDA per l'installazione di `flash-attn`.

```bash
pip install -r requirements.txt
```

Dipendenze principali: `torch`, `lightning`, `tokenizers`, `flash-attn`, `datasets`, `wandb`, `transformers`, `safetensors`.

### `requirements-vm.txt` - Ambiente VM (Fasi 1-2, solo CPU)

Dipendenze minime per l'esecuzione delle Fasi 1-2 (tokenizer + data pipeline) su VM cloud senza GPU. Esclude `torch`, `lightning`, `flash-attn` e altre librerie GPU-specifiche che non servono per il preprocessing dei dati.

```bash
pip install -r requirements-vm.txt
```

Dipendenze: `tokenizers`, `datasets`, `datasketch`, `numpy`, `tqdm`, `xxhash`, `pyyaml`.

Questo file viene utilizzato automaticamente dallo script `scripts/vm_setup.sh` durante il provisioning della VM Hetzner.

---

## Pipeline di Addestramento

L'addestramento completo si articola in 5 fasi sequenziali, ciascuna eseguibile tramite il rispettivo script.

### Fase 1 -- Addestramento Tokenizer

Addestra un tokenizer BPE da 32K token su un sottoinsieme di 5GB di CulturaX italiano.

```bash
bash scripts/01_train_tokenizer.sh
```

### Fase 2 -- Pipeline Dati

Scarica CulturaX italiano, filtra e deduplica i documenti, quindi pre-tokenizza il corpus in formato binario per il training.

```bash
bash scripts/02_preprocess_data.sh
```

### Fase 2.5 -- Smoke Test su GPU Locale

Verifica che l'intera pipeline di training funzioni correttamente sui dati reali addestrando il modello nano (11M parametri) per 200 step su GPU locale. Questo passo e' gratuito e permette di individuare problemi prima di pagare l'H100.

```bash
bash scripts/02_5_smoke_test.sh
```

Requisiti: 1x GPU con almeno 4GB VRAM (RTX 3060 12GB e' sufficiente). Tempo stimato: ~10-15 minuti.

### Fase 3 -- Pre-training

Addestra il modello da 1B parametri su 5 miliardi di token con ottimizzatore AdamW, schedule cosine con warmup e precisione BF16 mista.

```bash
bash scripts/03_pretrain.sh
```

### Fase 4 -- Fine-tuning

Instruction fine-tuning sul modello pre-addestrato per migliorare le capacita' di risposta a domande e istruzioni.

```bash
bash scripts/04_finetune.sh
```

### Fase 5 -- Valutazione

Esegue test di generazione e Q&A per valutare la qualita' del modello addestrato.

```bash
bash scripts/05_evaluate.sh
```

---

## Inferenza

### Generazione libera

```bash
python -m src.inference.generate \
    --model-dir checkpoints/finetune \
    --tokenizer-path tokenizer/boccaccio-32k.json \
    --config-path configs/model.yaml \
    --mode generate \
    --prompt "L'intelligenza artificiale in Italia" \
    --max-new-tokens 256 \
    --temperature 0.7
```

### Modalita' Q&A

```bash
python -m src.inference.generate \
    --model-dir checkpoints/finetune \
    --tokenizer-path tokenizer/boccaccio-32k.json \
    --config-path configs/model.yaml \
    --mode qa \
    --context "Roma e' la capitale della Repubblica Italiana. Si trova nella regione Lazio, nell'Italia centrale." \
    --question "Qual e' la capitale dell'Italia?" \
    --max-new-tokens 128 \
    --temperature 0.3
```

Parametri di generazione disponibili: `--temperature`, `--top-k`, `--top-p`, `--max-new-tokens`, `--device`.

---

## Configurazione

Tutti gli iperparametri sono centralizzati in file YAML nella directory `configs/`:

- **`configs/model.yaml`** -- Architettura del modello (dimensioni, numero di layer, GQA, ecc.). Include anche la variante `nano` per test locali rapidi.
- **`configs/tokenizer.yaml`** -- Parametri del tokenizer BPE (vocabolario, normalizzazione, corpus sorgente).
- **`configs/training.yaml`** -- Iperparametri di addestramento separati per pre-training e fine-tuning (learning rate, batch size, scheduler, checkpoint, hardware).

---

## Costi Stimati

| Risorsa          | Stima                              |
|------------------|------------------------------------|
| Hardware         | 1x NVIDIA H100 80GB                |
| Tempo pre-training | ~15-20 ore                       |
| Costo compute    | ~30-35 EUR                         |

Le stime si riferiscono al pre-training su 5 miliardi di token con le configurazioni di default. Il fine-tuning richiede risorse significativamente inferiori.

---

## Licenza

De Lauretis Tech -- Tutti i diritti riservati.
