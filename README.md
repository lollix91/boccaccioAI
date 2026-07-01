# BoccaccioAI

**Modello linguistico nativo italiano da 700 milioni di parametri, addestrato da zero.**

*De Lauretis Tech*

---

## Panoramica

BoccaccioAI e' un modello linguistico decoder-only da circa 700 milioni di parametri, addestrato interamente da zero su testo italiano estratto dal dataset CulturaX. Il modello e' ottimizzato per compiti di generazione testuale e Question & Answering in lingua italiana.

Caratteristiche principali:

- Tokenizer BPE custom con vocabolario da 32.000 token, addestrato specificamente su testo italiano.
- Grouped Query Attention (GQA) con rapporto 3:1 per ridurre la memoria in fase di inferenza.
- Funzione di attivazione SwiGLU per una migliore efficienza di addestramento.
- Rotary Position Embeddings (RoPE) per la codifica posizionale relativa.
- Compatibilita' con Flash Attention 2 per l'accelerazione delle operazioni di attenzione.
- Architettura scalata secondo le Chinchilla Scaling Laws (ratio token/parametri ~14:1).
- Fine-tuning con instruction following + refusal training (honesty) su 270k esempi italiani.

---

## Architettura

| Componente         | Specifica                          |
|--------------------|------------------------------------|
| Parametri          | ~700M                              |
| Hidden Size        | 1536                               |
| Layer              | 26                                 |
| Attention Heads    | 12 (GQA 3:1, 4 KV heads)          |
| FFN                | SwiGLU (intermediate 4096)         |
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
|   |-- 03_pretrain.sh          # Fase 3: pre-training modello 700M
|   |-- 04_finetune.sh          # Fase 4: instruction fine-tuning
|   |-- 05_evaluate.sh          # Fase 5: valutazione e test
|   |-- vm_setup.sh             # Setup VM Hetzner per Fasi 1-2
|   |-- run_fases_1_2.sh        # Esecuzione Fasi 1-2 su VM Hetzner
|   |-- lightning_setup.py      # Setup Studio Lightning.ai + avvio training
|   |-- lightning_monitor.py    # Monitoraggio training su Lightning.ai
|   |-- lightning_download.py   # Download checkpoint da Lightning.ai
|   |-- test_inference.py       # Test inferenza in locale con checkpoint Lightning
|   |-- vast_setup.py           # Setup Vast.ai + avvio pre-training (H100)
|   |-- vast_finetune.py        # Setup Vast.ai + avvio fine-tuning (H100)
|   |-- vast_monitor.py         # Monitoraggio training su Vast.ai
|   |-- auto_upload_ckpt.sh     # Daemon auto-upload checkpoint pretrain su HF Hub
|   |-- auto_upload_finetune.sh # Daemon auto-upload checkpoint finetune su HF Hub (generato da vast_finetune.py)
|   |-- check_hf.py             # Verifica file e spazio su HF Hub
|   |-- cleanup_hf.py           # Pulizia checkpoint intermedi su HF Hub
|   |-- upload_finetune.py      # Upload dataset finetune su HF Hub
|   |-- read_tb.py              # Lettura metriche TensorBoard da remoto
|-- src/
|   |-- data/
|   |   |-- download.py         # Download CulturaX italiano
|   |   |-- filter.py           # Filtraggio e deduplicazione documenti
|   |   |-- tokenize_corpus.py  # Pre-tokenizzazione in formato binario
|   |   |-- dataset.py          # Dataset PyTorch per il training
|   |   |-- prepare_finetune.py # Preparazione dataset fine-tuning (unione + tokenizzazione)
|   |   |-- generate_refusals.py # Generazione esempi refusal training (honesty)
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

Scarica CulturaX italiano (30GB), filtra e deduplica i documenti, quindi pre-tokenizza il corpus in formato binario per il training.

La pipeline di filtering e' composta da tre stage streaming (shard per shard) per funzionare su VM con RAM limitata (16GB):

1. **Heuristic filtering** - rimuove documenti troppo corti/lunghi, con basso ratio alfabetico, punteggiatura eccessiva, o linee ripetute. Output: `data/heuristic/`
2. **Exact deduplication (xxhash)** - rimuove duplicati esatti calcolando xxhash del testo normalizzato. RAM: ~200MB per 10M documenti. Output: `data/filtered/`
3. **Perplexity filtering** (opzionale) - richiede modello KenLM. Skippato se non fornito.

> **Nota sulla strategia di dedup**: Inizialmente era previsto MinHash LSH per catturare anche i near-duplicate, ma su 9.26M documenti con 128 permutazioni richiedeva ~15 ore e ~10GB di RAM, con rischio OOM su VM 16GB. L'exact dedup via xxhash completa in ~10 minuti con ~200MB di RAM. CulturaX e' gia' pre-deduplicato da HuggingFace, quindi i near-duplicate residui sono trascurabili per un modello da 700M parametri.

La **pre-tokenizzazione** converte il testo filtrato in token IDs binari (uint16) usando il tokenizer BPE addestrato in Fase 1. Anche questo step usa un'architettura streaming: tokenizza uno shard alla volta e scrive i token IDs direttamente su file binario, evitando di caricare l'intero corpus in RAM. Output: `data/tokenized/pretrain/train.bin`, `val.bin`, `meta.json`.

> **Nota sulla tokenizzazione streaming**: La versione originale caricava tutti i 30GB di testo + ~10B token IDs in RAM come liste Python, causando OOM su VM 16GB. La versione streaming usa ~240MB di RAM e completa in ~3 ore.

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

Addestra il modello da 700M parametri su 10 miliardi di token con ottimizzatore AdamW, schedule cosine con warmup e precisione BF16 mista. Il ratio token/parametri (~14:1) e' vicino all'ottimale secondo le Chinchilla Scaling Laws.

```bash
bash scripts/03_pretrain.sh
```

### Fase 4 -- Instruction Fine-tuning

Instruction fine-tuning sul modello pre-addestrato per migliorare le capacita' di risposta a domande e istruzioni. Il fine-tuning trasforma il modello da "completatore di testo" a "assistente che risponde a domande".

**Prerequisiti**:
- Checkpoint pre-training completato (`checkpoints/pretrain/last.ckpt`)
- Dataset instruction tokenizzato (`data/tokenized/finetune/train.bin`, `val.bin`)

**Dataset**: il dataset di fine-tuning e' composto da 3 fonti unite e tokenizzate in formato binario:

| Fonte | Esempi | Tipo | Descrizione |
|-------|--------|------|-------------|
| `anakin87/fine-instructions-ita-70k` | 69.890 | Instruction generali | Traduzione LLM-aided con quality filtering (giudice LLM) |
| `raicrits/Orca_ITA_200k` | 199.922 | Instruction + reasoning | Orca-style, system prompt + domande con ragionamento implicito |
| Refusal examples (generati) | 1.140 | Honesty training | Domande su eventi futuri, persone sconosciute, dati sensibili -> "Non ho informazioni sufficienti" |
| **Totale** | **270.952** | | **~102M token** |

Il **refusal training** insegna al modello a rispondere onestamente "Non ho informazioni sufficienti per rispondere a questa domanda" quando non conosce la risposta, invece di allucinare. Gli esempi coprono 6 categorie: eventi futuri, persone sconosciute/private, dettagli troppo specifici, domande mediche personali, domande legali specifiche, dati sensibili/privacy.

**Formato chat**: ogni esempio viene formattato come:
```
user <domanda> assistant <risposta> <|end|>
```

**Preparazione dataset** (in locale, non su GPU):
```bash
python src/data/generate_refusals.py     # Genera 1.140 esempi refusal
python src/data/prepare_finetune.py      # Scarica, unisce, tokenizza, salva in binario
python scripts/upload_finetune.py        # Upload su HF Hub
```

**Training**: ~45 min su H100 (102M token, 3 epoch, LR 2e-5, cosine decay, effective batch 65k token/step). Il training usa `PreTokenizedDataset` (formato binario, come il pre-training) invece di `InstructionDataset` (JSONL) per efficienza.

```bash
# Setup completo su Vast.ai (download dati + avvio training + daemon upload)
python scripts/vast_finetune.py --host <ip> --port <porta> --key ~/.ssh/vast_rsa
```

Il daemon `auto_upload_finetune.sh` monitora la directory `checkpoints/finetune/` e carica automaticamente ogni nuovo checkpoint su HF Hub, mantenendo solo gli ultimi 2 locali.

Dopo il fine-tuning, il modello puo' essere usato in modalita' chat/Q&A e risponde onestamente quando non conosce una risposta.

### Fase 5 -- Valutazione

Esegue test di generazione e Q&A per valutare la qualita' del modello addestrato.

```bash
bash scripts/05_evaluate.sh
```

---

## Test Inferenza in Locale

Lo script `scripts/test_inference.py` permette di testare un checkpoint Lightning (`.ckpt`) su GPU locale (es. RTX 3060 12GB). A differenza di `src/inference/generate.py`, carica direttamente i checkpoint di PyTorch Lightning senza dover esportare in `model.pt`.

### Generazione libera (pre-train)

```bash
python scripts/test_inference.py \
    --checkpoint checkpoints/pretrain/last.ckpt \
    --tokenizer-path tokenizer/boccaccio-32k.json \
    --prompt "L'intelligenza artificiale in Italia" \
    --max-new-tokens 256 \
    --temperature 0.7
```

### Modalita' Q&A (dopo fine-tuning)

```bash
python scripts/test_inference.py \
    --checkpoint checkpoints/finetune/last.ckpt \
    --tokenizer-path tokenizer/boccaccio-32k.json \
    --mode qa \
    --context "Roma e' la capitale della Repubblica Italiana." \
    --question "Qual e' la capitale dell'Italia?" \
    --max-new-tokens 128 \
    --temperature 0.3
```

### Parametri disponibili

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--checkpoint` | (richiesto) | Path al file .ckpt |
| `--tokenizer-path` | `tokenizer/boccaccio-32k.json` | Path al tokenizer |
| `--config-path` | `configs/model.yaml` | Config del modello |
| `--config-variant` | `model` | Variante (model o nano) |
| `--mode` | `generate` | generate o qa |
| `--prompt` | - | Prompt per generate |
| `--context` | - | Contesto per qa |
| `--question` | - | Domanda per qa |
| `--max-new-tokens` | 256 | Token da generare |
| `--temperature` | 0.7 | Temperatura sampling (0 = greedy) |
| `--top-k` | 50 | Top-k filtering |
| `--top-p` | 0.9 | Top-p nucleus sampling |
| `--device` | auto | cuda, cuda:0, cpu |

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

### Configurazione fine-tuning

Il file `configs/training.yaml` sezione `finetune` contiene:

| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| `micro_batch_size` | 8 | Batch per-GPU (H100 80GB) |
| `gradient_accumulation_steps` | 4 | Accumulazione gradienti |
| Effective batch | 65.536 token/step | 8 * 4 * 2048 |
| `learning_rate` | 2e-5 | LR basso per fine-tuning |
| `min_learning_rate` | 2e-6 | LR finale (cosine decay) |
| `warmup_steps` | 50 | Warmup breve |
| `num_epochs` | 3 | Epoch totali |
| `num_tokens` | 102M | Token dataset (per cosine schedule) |
| `compile_model` | false | torch.compile disabilitato (overhead non giustificato per training breve) |
| `use_flash_attention` | true | Flash Attention 2 |
| `checkpoint_every_n_steps` | 1500 | Salvataggio ogni ~1500 step |
| `val_check_interval` | 200 | Validation ogni 200 step |

---

## Infrastruttura Vast.ai

Il training avviene su istanze Vast.ai con GPU H100 80GB. Gli script automatizzano il setup completo:

### Pre-training (`scripts/vast_setup.py`)
1. Verifica GPU (H100/A100/RTX)
2. Clone/pull del repository
3. Download dati + checkpoint da HF Hub
4. Creazione daemon auto-upload checkpoint
5. Avvio training in tmux + daemon in tmux separato

### Fine-tuning (`scripts/vast_finetune.py`)
1. Stesso setup del pre-training
2. Download dataset fine-tuning + checkpoint pretrain da HF
3. Daemon auto-upload specifico per `checkpoints/finetune/`
4. Avvio training con `--mode finetune --resume-from checkpoints/pretrain/last.ckpt`

### Monitoraggio (`scripts/vast_monitor.py`)
- Stato tmux sessions
- Ultime righe del log di training
- Utilizzo GPU (nvidia-smi)
- Lista checkpoint locali
- Metriche TensorBoard (tramite `scripts/read_tb.py`)

### Daemon auto-upload
Il daemon monitora la directory dei checkpoint e carica automaticamente ogni nuovo file su HF Hub. Mantiene un registry dei file gia' caricati (`.uploaded_*_registry`) ed elimina i checkpoint locali piu' vecchi per risparmiare spazio su disco. Il token HF viene letto dal file `.hf_token` sul server.

---

## Costi Stimati

| Risorsa          | Stima                              |
|------------------|------------------------------------|
| Hardware         | 1x NVIDIA H100 80GB (Vast.ai)     |
| Tempo pre-training | ~25 ore (12.779 step, 6.7B token) |
| Costo pre-training | ~$48 (Vast.ai H100 a $1.93/h)    |
| Tempo fine-tuning | ~45 min (102M token, 3 epoch)     |
| Costo fine-tuning | ~$1.5                            |

Le stime si riferiscono al pre-training su 6.7 miliardi di token (1 epoch) con le configurazioni di default. Il fine-tuning richiede risorse significativamente inferiori.

---

## Stato di Avanzamento

| Fase | Stato | Dettagli |
|------|-------|----------|
| 1. Tokenizer | Completato | BPE 32K su 5GB CulturaX italiano |
| 2. Pipeline dati | Completato | 30GB CulturaX, filtering + dedup, 6.7B token |
| 2.5. Smoke test | Completato | Modello nano su RTX 3060 locale |
| 3. Pre-training | Completato | 12.779 step su H100, train loss 2.55, val loss 3.26 |
| 4. Fine-tuning | In corso | 270k esempi (70k instruction + 200k Orca + 1.1k refusal), 3 epoch su H100 |
| 5. Valutazione | Da fare | Test inferenza locale dopo fine-tuning |

**Storage HuggingFace Hub** (`lollix91/boccaccio-data`): ~22 GB totali
- Checkpoint pre-training `last.ckpt` (8.4 GB)
- Dati pre-training tokenizzati `train.bin` + `val.bin` (13.3 GB)
- Dati fine-tuning tokenizzati `train.bin` + `val.bin` (204 MB)
- Tokenizer, config, meta.json

**Repository GitHub**: `lollix91/boccaccioAI` -- codice sorgente, script, configurazione. I token API sono gestiti tramite variabili d'ambiente (`HF_TOKEN`) e file `.hf_token` sul server remoto.

---

## Licenza

De Lauretis Tech -- Tutti i diritti riservati.
