"""BoccaccioAI - VM Progress Monitor.

Si collega alla VM via SSH e mostra lo stato di avanzamento
delle Fasi 1-2. Legge progress.json e i log della pipeline.

Uso:
    python scripts/vm_monitor.py --host <IP> --password <PASSWORD>

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="BoccaccioAI - VM Progress Monitor")
    parser.add_argument("--host", required=True, help="VM IP address")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--user", default="root", help="SSH user")
    parser.add_argument("--password", default=None, help="SSH password")
    parser.add_argument("--tail", type=int, default=20, help="Numero righe di log da mostrare")
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

    # ─── Leggi progress.json ──────────────────────────────
    code, progress_raw = run_command(ssh, "cat /opt/boccaccioAI/progress.json 2>/dev/null")

    if code != 0 or not progress_raw:
        print("Nessun file di progresso trovato.")
        if tmux_running:
            print("La sessione tmux e' attiva ma il progresso non e' ancora stato scritto.")
            print("Il pipeline potrebbe essere ancora in fase di avvio.")
        else:
            print("La sessione tmux non e' attiva. Il pipeline potrebbe non essere stato avviato.")
        ssh.close()
        sys.exit(0)

    try:
        progress = json.loads(progress_raw)
    except json.JSONDecodeError:
        print(f"Errore parsing progress.json: {progress_raw}")
        ssh.close()
        sys.exit(1)

    # ─── Mostra stato ─────────────────────────────────────
    stage = progress.get("stage", "unknown")
    stage_name = progress.get("stage_name", "Sconosciuto")
    percent = progress.get("percent", 0)
    status = progress.get("status", "unknown")
    elapsed = progress.get("elapsed_seconds", 0)

    print(f"  Stato:      {'RUNNING' if tmux_running else 'STOPPED'}")
    print(f"  Fase:       {STAGE_EMOJI.get(stage, '[?]')} {stage_name}")
    print(f"  Progresso:  {percent}%")
    print(f"  Tempo:      {format_elapsed(elapsed)}")
    print(f"  Aggiornato: {progress.get('updated_at', 'N/A')}")
    print()

    # ─── Barra di progresso ───────────────────────────────
    bar_width = 40
    filled = int(bar_width * percent / 100)
    bar = "=" * filled + "-" * (bar_width - filled)
    print(f"  [{bar}] {percent}%")
    print()

    # ─── Stato pipeline ───────────────────────────────────
    print("  Stato fasi:")
    for s, order in sorted(STAGE_ORDER.items(), key=lambda x: x[1]):
        emoji = STAGE_EMOJI.get(s, "[?]")
        name_map = {
            "fase_1_tokenizer": "Tokenizer BPE 32K",
            "fase_2a_download": "Download CulturaX IT",
            "fase_2b_filtering": "Filtering e Dedup",
            "fase_2c_tokenize": "Pre-tokenizzazione",
            "completed": "Completato",
        }
        if stage == s and status == "running":
            print(f"    {emoji} {name_map.get(s, s):30s} >>> IN ESECUZIONE")
        elif STAGE_ORDER.get(stage, 0) > order or stage == "completed":
            print(f"    {emoji} {name_map.get(s, s):30s}     COMPLETATO")
        elif stage == s and status == "completed":
            print(f"    {emoji} {name_map.get(s, s):30s}     COMPLETATO")
        else:
            print(f"    {emoji} {name_map.get(s, s):30s}     IN ATTESA")

    print()

    # ─── Ultime righe di log ──────────────────────────────
    if args.tail > 0:
        print(f"  Ultime {args.tail} righe di log:")
        print("  " + "-" * 56)
        code, log_out = run_command(
            ssh,
            f"tail -n {args.tail} /opt/boccaccioAI/logs/full_pipeline.log 2>/dev/null"
        )
        if log_out:
            for line in log_out.split("\n"):
                print(f"  {line}")
        else:
            print("  (nessun log disponibile)")
        print()

    # ─── Dimensioni file output ───────────────────────────
    print("  File generati:")
    files_to_check = [
        ("tokenizer/boccaccio-32k.json", "Tokenizer"),
        ("data/tokenized/pretrain/train.bin", "Dati train"),
        ("data/tokenized/pretrain/val.bin", "Dati val"),
        ("data/tokenized/pretrain/meta.json", "Metadata"),
    ]
    for fpath, label in files_to_check:
        code, out = run_command(ssh, f"ls -lh /opt/boccaccioAI/{fpath} 2>/dev/null | awk '{{print $5}}'")
        if code == 0 and out:
            print(f"    {label:20s} {out}")
        else:
            print(f"    {label:20s} (non ancora creato)")

    print()
    print("=" * 60)

    if stage == "completed":
        print("  FASI 1-2 COMPLETATE!")
        print("  Esegui: python scripts/vm_download.py per scaricare i risultati")
        print("=" * 60)

    ssh.close()


if __name__ == "__main__":
    main()
