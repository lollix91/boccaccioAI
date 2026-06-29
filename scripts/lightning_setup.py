"""BoccaccioAI - Lightning.ai Studio Setup & Training (HF Hub edition).

Si collega allo Studio via SSH, scarica i dati da HuggingFace Hub,
verifica le dipendenze, e lancia il training in una sessione tmux persistente.

Uso:
    python scripts/lightning_setup.py
    python scripts/lightning_setup.py --skip-download  # se dati gia' scaricati

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import paramiko


# ─── Configurazione Lightning.ai ──────────────────────────────

LIGHTNING_HOST = "ssh.lightning.ai"
LIGHTNING_PORT = 22
LIGHTNING_USER = "s_01kw9jgs29f9znwd4cwpcctbpa"
LIGHTNING_KEY = os.path.expanduser("~/.ssh/lightning_rsa")

CONDA_PYTHON = "/home/zeus/miniconda3/envs/cloudspace/bin/python"
CONDA_PIP = "/home/zeus/miniconda3/envs/cloudspace/bin/pip"
PROJECT_DIR = "/home/zeus/content/boccaccioAI"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "lollix91/boccaccio-data"


# ─── Helper ───────────────────────────────────────────────────

def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 300) -> tuple[int, str]:
    """Esegue un comando SSH e ritorna (exit_code, output)."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_code, (out + err).strip()


# ─── Step 1: Verifica GPU ─────────────────────────────────────

def check_gpu(ssh: paramiko.SSHClient) -> bool:
    print()
    print("=== Step 1: Verifica GPU ===")
    code, out = run(ssh, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null")
    if code != 0 or not out:
        print("  ERRORE: GPU non disponibile.")
        return False
    print(f"  GPU: {out}")
    return "H100" in out


# ─── Step 2: Setup progetto e dipendenze ──────────────────────

def setup_project(ssh: paramiko.SSHClient) -> bool:
    print()
    print("=== Step 2: Setup progetto ===")

    code, out = run(ssh, f"test -d {PROJECT_DIR}/.git && echo exists")
    if "exists" in out:
        print("  Repo presente, faccio pull...")
        run(ssh, f"cd {PROJECT_DIR} && git pull origin main 2>&1")
    else:
        print("  Clonando il repo...")
        code, out = run(ssh, f"cd /home/zeus/content && git clone https://github.com/lollix91/boccaccioAI.git 2>&1")
        if code != 0:
            print(f"  ERRORE: {out}")
            return False

    print("  Verifico dipendenze...")
    deps = ["lightning", "tokenizers", "tqdm", "pyyaml", "numpy", "xxhash", "huggingface_hub"]
    missing = []
    for dep in deps:
        code, out = run(ssh, f"{CONDA_PYTHON} -c 'import {dep}' 2>/dev/null && echo OK || echo MISSING")
        if "MISSING" in out:
            missing.append(dep)
        else:
            print(f"    {dep}: OK")

    if missing:
        print(f"  Installo: {', '.join(missing)}")
        run(ssh, f"{CONDA_PIP} install {' '.join(missing)} 2>&1 | tail -3", timeout=180)

    print("  Verifico PyTorch + CUDA...")
    code, out = run(ssh, f'{CONDA_PYTHON} -c "import torch; print(f\'torch={{torch.__version__}} cuda={{torch.cuda.is_available()}} gpu={{torch.cuda.get_device_name(0)}}\')"')
    print(f"    {out}")
    return "cuda=True" in out


# ─── Step 3: Download dati da HF Hub ──────────────────────────

def download_data(ssh: paramiko.SSHClient) -> bool:
    print()
    print("=== Step 3: Download dati da HuggingFace Hub ===")
    print(f"  Repo: {HF_REPO}")
    print("  Questo richiede ~5-10 min (bandwidth HF e' alta).")
    print()

    # Rimuovi file parziali dal vecchio upload SFTP
    print("  Rimuovo file parziali dal vecchio upload SFTP...")
    run(ssh, f"rm -f {PROJECT_DIR}/data/tokenized/pretrain/train.bin {PROJECT_DIR}/data/tokenized/pretrain/val.bin {PROJECT_DIR}/data/tokenized/pretrain/meta.json {PROJECT_DIR}/tokenizer/boccaccio-32k.json 2>/dev/null")

    # Script Python: usa snapshot_download con local_dir = project root
    # per preservare la struttura del repo
    download_script = f"""
import os
from huggingface_hub import snapshot_download

os.environ["HF_TOKEN"] = "{HF_TOKEN}"
repo = "{HF_REPO}"
project = "{PROJECT_DIR}"

print(f"Downloading snapshot to {{project}}...")
snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    local_dir=project,
    token=os.environ["HF_TOKEN"],
)
print("=== Verifica file ===")
for f in ["data/tokenized/pretrain/train.bin", "data/tokenized/pretrain/val.bin",
          "data/tokenized/pretrain/meta.json", "tokenizer/boccaccio-32k.json"]:
    path = os.path.join(project, f)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f"OK: {{f}} ({{size_mb:.1f}} MB)")
    else:
        print(f"MISSING: {{f}}")
print("=== Download completo ===")
"""

    run(ssh, f"mkdir -p {PROJECT_DIR}/scripts")
    run(ssh, f"cat > {PROJECT_DIR}/scripts/_hf_download.py << 'PYEOF'\n{download_script}\nPYEOF")
    code, out = run(ssh, f"{CONDA_PYTHON} {PROJECT_DIR}/scripts/_hf_download.py 2>&1", timeout=900)
    # Sanitize per evitare UnicodeEncodeError su Windows (cp1252)
    safe = out.encode("ascii", errors="replace").decode("ascii")
    # Mostra solo le ultime righe (i progress bar spammano)
    lines = [l for l in safe.split("\n") if l.strip()]
    for line in lines[-20:]:
        print(f"  {line}")

    run(ssh, f"rm {PROJECT_DIR}/scripts/_hf_download.py 2>/dev/null")

    return "Download completo" in out


# ─── Step 4: Verifica dati ────────────────────────────────────

def verify_data(ssh: paramiko.SSHClient) -> bool:
    print()
    print("=== Step 4: Verifica dati ===")

    checks = [
        ("tokenizer/boccaccio-32k.json", 2_000_000),
        ("data/tokenized/pretrain/train.bin", 13_000_000_000),
        ("data/tokenized/pretrain/val.bin", 60_000_000),
        ("data/tokenized/pretrain/meta.json", 100),
    ]

    all_ok = True
    for rel, min_size in checks:
        path = f"{PROJECT_DIR}/{rel}"
        code, out = run(ssh, f"stat -c '%s' {path} 2>/dev/null")
        if code != 0 or not out:
            print(f"  MANCANTE: {rel}")
            all_ok = False
            continue
        size = int(out)
        mb = size / 1e6
        if size < min_size:
            print(f"  INCOMPLETO: {rel} ({mb:.1f} MB)")
            all_ok = False
        else:
            print(f"  OK: {rel} ({mb:.1f} MB)")
    return all_ok


# ─── Step 5: Avvia training in tmux ───────────────────────────

def start_training(ssh: paramiko.SSHClient) -> bool:
    print()
    print("=== Step 5: Avvio training ===")

    run(ssh, "tmux kill-session -t boccaccio 2>/dev/null")
    run(ssh, f"mkdir -p {PROJECT_DIR}/checkpoints/pretrain {PROJECT_DIR}/logs")

    train_cmd = (
        f"cd {PROJECT_DIR} && "
        f"{CONDA_PYTHON} -m src.training.train "
        "--mode pretrain "
        "--model-config configs/model.yaml "
        "--model-variant model "
        "--training-config configs/training.yaml "
        "--data-path data/tokenized/pretrain "
        "--tokenizer-path tokenizer/boccaccio-32k.json "
        "--wandb-offline "
        "2>&1 | tee logs/pretrain_training.log"
    )

    code, out = run(ssh, f"tmux new-session -d -s boccaccio '{train_cmd}'")
    if code != 0:
        print(f"  ERRORE tmux: {out}")
        return False

    time.sleep(8)
    code, out = run(ssh, "tmux list-sessions 2>/dev/null")
    if "boccaccio" not in out:
        print("  ERRORE: tmux non avviato.")
        code, log = run(ssh, f"tail -20 {PROJECT_DIR}/logs/pretrain_training.log 2>/dev/null")
        if log:
            print(f"  Log: {log}")
        return False

    print("  Training avviato in tmux 'boccaccio'")
    return True


# ─── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BoccaccioAI - Lightning.ai Setup (HF Hub)")
    parser.add_argument("--host", default=LIGHTNING_HOST)
    parser.add_argument("--port", type=int, default=LIGHTNING_PORT)
    parser.add_argument("--user", default=LIGHTNING_USER)
    parser.add_argument("--key", default=LIGHTNING_KEY)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  BoccaccioAI - Lightning.ai Setup & Training")
    print("=" * 60)

    print(f"\nConnessione a {args.host}...", end=" ")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(args.host, port=args.port, username=args.user, key_filename=args.key, timeout=15)
    except Exception as e:
        print(f"FALLITO: {e}")
        sys.exit(1)
    print("OK")

    if not check_gpu(ssh):
        ssh.close(); sys.exit(1)

    if not setup_project(ssh):
        ssh.close(); sys.exit(1)

    if not args.skip_download:
        if not download_data(ssh):
            ssh.close(); sys.exit(1)
    else:
        print("\n=== Step 3: Download (saltato) ===")

    if not verify_data(ssh):
        ssh.close(); sys.exit(1)

    if not start_training(ssh):
        ssh.close(); sys.exit(1)

    ssh.close()

    print()
    print("=" * 60)
    print("  SETUP COMPLETATO - Training in corso!")
    print("=" * 60)
    print()
    print("  Puoi chiudere PowerShell: il training continua in tmux.")
    print()
    print("  Monitora:  python scripts/lightning_monitor.py")
    print("  Scarica:   python scripts/lightning_download.py")
    print()


if __name__ == "__main__":
    main()
