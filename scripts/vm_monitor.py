"""BoccaccioAI - VM Progress Monitor.

Si collega alla VM via SSH e mostra lo stato di avanzamento
delle Fasi 1-2. Legge progress.json, i log della pipeline,
e le dimensioni dei file su disco per determinare lo stato reale.

Uso:
    python scripts/vm_monitor.py --host <IP> --password <PASSWORD>

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import paramiko


def run_command(ssh: paramiko.SSHClient, cmd: str, timeout: int = 30) -> tuple[int, str]:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_code, (out + err).strip()


def format_elapsed(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def parse_download_progress(log_text: str) -> int | None:
    """Estrae la percentuale di download dal log di tqdm."""
    # Cerca l'ultima riga con "Downloading CulturaX IT: XX%"
    matches = re.findall(r"Downloading CulturaX IT:\s+(\d+)%", log_text)
    if matches:
        return int(matches[-1])
    return None


def detect_stage(ssh: paramiko.SSHClient) -> dict:
    """Rileva lo stage corrente analizzando processi e file su disco."""
    # Controlla quali processi Python stanno girando
    code, ps_out = run_command(
        ssh, "ps aux | grep -E 'src\\.data\\.(download|filter|tokenize)' | grep -v grep"
    )

    if "src.data.download" in ps_out:
        return {"stage": "fase_2a_download", "stage_name": "Download CulturaX IT", "status": "running"}
    if "src.data.filter" in ps_out:
        return {"stage": "fase_2b_filtering", "stage_name": "Filtering e Dedup", "status": "running"}
    if "src.data.tokenize" in ps_out:
        return {"stage": "fase_2c_tokenize", "stage_name": "Pre-tokenizzazione", "status": "running"}

    # Nessun processo attivo - controlla se tutto e' completato
    code, train_bin = run_command(ssh, "ls /opt/boccaccioAI/data/tokenized/pretrain/train.bin 2>/dev/null")
    if train_bin:
        return {"stage": "completed", "stage_name": "Fasi 1-2 completate", "status": "completed"}

    # Controlla cosa esiste per determinare lo stage
    code, raw_exists = run_command(ssh, "ls /opt/boccaccioAI/data/raw/*.jsonl 2>/dev/null | head -1")
    code, filtered_exists = run_command(ssh, "ls /opt/boccaccioAI/data/filtered/*.jsonl 2>/dev/null | head -1")

    if filtered_exists:
        return {"stage": "fase_2c_tokenize", "stage_name": "Pre-tokenizzazione", "status": "pending"}
    if raw_exists:
        return {"stage": "fase_2b_filtering", "stage_name": "Filtering e Dedup", "status": "pending"}

    return {"stage": "unknown", "stage_name": "Sconosciuto", "status": "stopped"}


STAGE_ORDER = {
    "fase_1_tokenizer": 1,
    "fase_2a_download": 2,
    "fase_2b_filtering": 3,
    "fase_2c_tokenize": 4,
    "completed": 5,
}

STAGE_EMOJI = {
    "fase_1_tokenizer": "[1]",
    "fase_2a_download": "[2]",
    "fase_2b_filtering": "[3]",
    "fase_2c_tokenize": "[4]",
    "completed": "[OK]",
}

STAGE_NAMES = {
    "fase_1_tokenizer": "Tokenizer BPE 32K",
    "fase_2a_download": "Download CulturaX IT",
    "fase_2b_filtering": "Filtering e Dedup",
    "fase_2c_tokenize": "Pre-tokenizzazione",
    "completed": "Completato",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="BoccaccioAI - VM Progress Monitor")
    parser.add_argument("--host", required=True, help="VM IP address")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--user", default="root", help="SSH user")
    parser.add_argument("--password", default=None, help="SSH password")
    parser.add_argument("--tail", type=int, default=15, help="Numero righe di log da mostrare")
    args = parser.parse_args()

    print("=" * 60)
    print("  BoccaccioAI - Monitoraggio Progresso VM")
    print("=" * 60)
    print()

    # ─── Connessione ──────────────────────────────────────
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(args.host, port=args.port, username=args.user,
                     password=args.password, timeout=15)
    except Exception as e:
        print(f"ERRORE: Impossibile connettersi a {args.host}: {e}")
        sys.exit(1)

    # ─── Verifica processo attivo ─────────────────────────
    code, tmux_out = run_command(ssh, "tmux list-sessions 2>/dev/null")
    tmux_running = "boccaccio" in tmux_out

    # ─── Rileva stage corrente dai processi ───────────────
    detected = detect_stage(ssh)
    stage = detected["stage"]
    stage_name = detected["stage_name"]
    status = detected["status"]

    # ─── Cerca percentuale di download dal log ────────────
    percent = 0
    if stage == "fase_2a_download":
        code, dl_log = run_command(
            ssh, "tail -100 /opt/boccaccioAI/logs/fase_2a_download.log 2>/dev/null"
        )
        parsed_pct = parse_download_progress(dl_log)
        if parsed_pct is not None:
            percent = parsed_pct
    elif stage == "completed":
        percent = 100
    elif status == "completed":
        percent = 100

    # ─── Mostra stato ─────────────────────────────────────
    print(f"  Stato:      {'RUNNING' if tmux_running else 'STOPPED'}")
    print(f"  Fase:       {STAGE_EMOJI.get(stage, '[?]')} {stage_name}")
    print(f"  Status:     {status}")
    if percent > 0:
        print(f"  Progresso:  {percent}%")
    print()

    # ─── Barra di progresso ───────────────────────────────
    if percent > 0:
        bar_width = 40
        filled = int(bar_width * percent / 100)
        bar = "=" * filled + "-" * (bar_width - filled)
        print(f"  [{bar}] {percent}%")
        print()

    # ─── Stato pipeline ───────────────────────────────────
    print("  Stato fasi:")
    for s, order in sorted(STAGE_ORDER.items(), key=lambda x: x[1]):
        emoji = STAGE_EMOJI.get(s, "[?]")
        name = STAGE_NAMES.get(s, s)
        if stage == s and status == "running":
            print(f"    {emoji} {name:30s} >>> IN ESECUZIONE")
        elif STAGE_ORDER.get(stage, 0) > order or stage == "completed":
            print(f"    {emoji} {name:30s}     COMPLETATO")
        elif stage == s and status == "pending":
            print(f"    {emoji} {name:30s}     IN ATTESA")
        else:
            print(f"    {emoji} {name:30s}     IN ATTESA")

    print()

    # ─── Ultime righe di log ──────────────────────────────
    if args.tail > 0:
        # Scegli il log giusto in base allo stage
        if stage == "fase_2a_download":
            log_file = "logs/fase_2a_download.log"
        elif stage == "fase_2b_filtering":
            log_file = "logs/fase_2b_filtering.log"
        elif stage == "fase_2c_tokenize":
            log_file = "logs/fase_2c_tokenize.log"
        else:
            log_file = "logs/full_pipeline.log"

        print(f"  Ultime {args.tail} righe di log ({log_file}):")
        print("  " + "-" * 56)
        code, log_out = run_command(
            ssh,
            f"tail -n {args.tail} /opt/boccaccioAI/{log_file} 2>/dev/null | cat -v | grep -v 'httpx\\|HTTP Request\\|Partial Content'"
        )
        if log_out:
            for line in log_out.split("\n"):
                # Tronca righe troppo lunghe (progress bar di tqdm)
                if len(line) > 120:
                    line = line[:120] + "..."
                print(f"  {line}")
        else:
            print("  (nessun log disponibile)")
        print()

    # ─── Dimensioni file su disco ─────────────────────────
    print("  File su disco:")
    dirs_to_check = [
        ("tokenizer/boccaccio-32k.json", "Tokenizer"),
        ("data/raw/", "Dati raw (download)"),
        ("data/filtered/", "Dati filtrati"),
        ("data/tokenized/pretrain/train.bin", "Dati train"),
        ("data/tokenized/pretrain/val.bin", "Dati val"),
        ("data/tokenized/pretrain/meta.json", "Metadata"),
    ]
    for fpath, label in dirs_to_check:
        if fpath.endswith("/"):
            code, out = run_command(ssh, f"du -sh /opt/boccaccioAI/{fpath} 2>/dev/null | awk '{{print $1}}'")
            if out and out != "0K" and out != "4.0K":
                print(f"    {label:25s} {out}")
            else:
                print(f"    {label:25s} (vuoto)")
        else:
            code, out = run_command(ssh, f"ls -lh /opt/boccaccioAI/{fpath} 2>/dev/null | awk '{{print $5}}'")
            if code == 0 and out:
                print(f"    {label:25s} {out}")
            else:
                print(f"    {label:25s} (non ancora creato)")

    print()
    print("=" * 60)

    if stage == "completed":
        print("  FASI 1-2 COMPLETATE!")
        print("  Esegui: python scripts/vm_download.py per scaricare i risultati")
        print("=" * 60)
    elif not tmux_running and status != "completed":
        print("  ATTENZIONE: la sessione tmux non e' attiva.")
        print("  Il pipeline potrebbe essersi fermato o aver crashato.")
        print("  Controlla i log per dettagli.")
        print("=" * 60)

    ssh.close()


if __name__ == "__main__":
    main()
