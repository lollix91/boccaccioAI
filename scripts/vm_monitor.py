"""BoccaccioAI - VM Progress Monitor.

Si collega alla VM via SSH e mostra lo stato di avanzamento
delle Fasi 1-2. Legge lo schermo di tmux (capture-pane) per
la progress bar reale di tqdm, e controlla i file su disco.

Uso:
    python scripts/vm_monitor.py --host <IP> --password <PASSWORD>

De Lauretis Tech
"""

from __future__ import annotations

import argparse
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


def parse_progress_from_tmux(tmux_text: str) -> tuple[str | None, int | None]:
    """Estrae stage e percentuale dallo schermo di tmux.

    Cerca righe come:
      Heuristic filter:  42%|####  | 26/62 [05:39<07:52, 13.13s/it]
      Building LSH index:  15%|##   | 3/20 [01:23<08:00, ...]
      Downloading CulturaX IT:  28%|###  | 8.5G/30.0G [03:04<07:30, ...]
      Tokenizing:  10%|#    | 5/50 [02:00<18:00, ...]
    """
    lines = tmux_text.split("\n")
    for line in reversed(lines):
        # Match generic tqdm progress: "Label: XX%|... | A/B [time<eta, ...]"
        m = re.search(r"(\w[\w ]+?):\s+(\d+)%.*?\|\s+(\d+)/(\d+)", line)
        if m:
            label = m.group(1).strip()
            percent = int(m.group(2))
            current = int(m.group(3))
            total = int(m.group(4))
            return label, percent

        # Match download progress: "Downloading ...: XX%|... | 8.5G/30.0G"
        m = re.search(r"(Downloading[\w ]+?):\s+(\d+)%", line)
        if m:
            return m.group(1).strip(), int(m.group(2))

    return None, None


def detect_stage_from_tmux(tmux_text: str) -> dict:
    """Rileva lo stage dal contenuto dello schermo tmux."""
    text_lower = tmux_text.lower()

    if "=== completed ===" in text_lower:
        return {"stage": "completed", "stage_name": "Completato", "status": "completed"}
    if "fase 2c" in text_lower or "tokeniz" in text_lower and "pre-token" in text_lower:
        return {"stage": "fase_2c_tokenize", "stage_name": "Pre-tokenizzazione", "status": "running"}
    if "building lsh" in text_lower or "dedup" in text_lower or "pass 2" in text_lower:
        return {"stage": "fase_2b_filtering", "stage_name": "Filtering - MinHash Dedup", "status": "running"}
    if "heuristic filter" in text_lower or "stage 1" in text_lower:
        return {"stage": "fase_2b_filtering", "stage_name": "Filtering - Heuristic", "status": "running"}
    if "downloading culturax" in text_lower or "fase 2a" in text_lower:
        return {"stage": "fase_2a_download", "stage_name": "Download CulturaX IT", "status": "running"}

    return {"stage": "unknown", "stage_name": "Sconosciuto", "status": "unknown"}


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

    # ─── Verifica tmux ────────────────────────────────────
    code, tmux_sessions = run_command(ssh, "tmux list-sessions 2>/dev/null")
    tmux_running = "boccaccio" in tmux_sessions

    # ─── Cattura schermo tmux ─────────────────────────────
    tmux_text = ""
    if tmux_running:
        code, tmux_text = run_command(
            ssh, "tmux capture-pane -t boccaccio -p -S -30 2>/dev/null"
        )

    # ─── Rileva stage e percentuale ───────────────────────
    if tmux_text:
        detected = detect_stage_from_tmux(tmux_text)
        label, percent = parse_progress_from_tmux(tmux_text)
    else:
        detected = {"stage": "unknown", "stage_name": "Sconosciuto", "status": "stopped"}
        label, percent = None, None

    # Se tmux non attivo, controlla file su disco per stato
    if not tmux_running:
        code, train_bin = run_command(ssh, "ls /opt/boccaccioAI/data/tokenized/pretrain/train.bin 2>/dev/null")
        if train_bin:
            detected = {"stage": "completed", "stage_name": "Completato", "status": "completed"}
            percent = 100
        else:
            code, filtered = run_command(ssh, "ls /opt/boccaccioAI/data/filtered/*.jsonl 2>/dev/null | head -1")
            if filtered:
                detected = {"stage": "fase_2c_tokenize", "stage_name": "Pre-tokenizzazione", "status": "pending"}
            code, heuristic = run_command(ssh, "ls /opt/boccaccioAI/data/heuristic/*.jsonl 2>/dev/null | head -1")
            if heuristic and detected["stage"] == "unknown":
                detected = {"stage": "fase_2b_filtering", "stage_name": "Filtering e Dedup", "status": "pending"}

    stage = detected["stage"]
    stage_name = detected["stage_name"]
    status = detected["status"]

    # ─── Mostra stato ─────────────────────────────────────
    print(f"  Stato:      {'RUNNING' if tmux_running else 'STOPPED'}")
    print(f"  Fase:       {STAGE_EMOJI.get(stage, '[?]')} {stage_name}")
    print(f"  Status:     {status}")
    if percent is not None:
        print(f"  Progresso:  {percent}%")
    print()

    # ─── Barra di progresso ───────────────────────────────
    if percent is not None and percent > 0:
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

    # ─── Schermo tmux (ultime righe utili) ────────────────
    if tmux_running and tmux_text:
        # Filtra righe vuote e mostra ultime 10 significative
        lines = [l for l in tmux_text.split("\n") if l.strip()]
        useful = []
        for line in lines:
            # Salta righe di log HTTP
            if "httpx" in line or "HTTP Request" in line:
                continue
            useful.append(line)

        print(f"  Schermo tmux (ultime {min(10, len(useful))} righe):")
        print("  " + "-" * 56)
        for line in useful[-10:]:
            # Pulisci caratteri di controllo e tronca
            clean = line.replace("\r", "").strip()
            if len(clean) > 120:
                clean = clean[:120] + "..."
            print(f"  {clean}")
        print()

    # ─── Dimensioni file su disco ─────────────────────────
    print("  File su disco:")
    dirs_to_check = [
        ("tokenizer/boccaccio-32k.json", "Tokenizer", "file"),
        ("data/raw/", "Dati raw (download)", "dir"),
        ("data/heuristic/", "Dati heuristic (interm.)", "dir"),
        ("data/filtered/", "Dati filtrati", "dir"),
        ("data/tokenized/pretrain/train.bin", "Dati train", "file"),
        ("data/tokenized/pretrain/val.bin", "Dati val", "file"),
        ("data/tokenized/pretrain/meta.json", "Metadata", "file"),
    ]
    for fpath, label, ftype in dirs_to_check:
        if ftype == "dir":
            code, out = run_command(ssh, f"du -sh /opt/boccaccioAI/{fpath} 2>/dev/null | awk '{{print $1}}'")
            code2, count = run_command(ssh, f"ls /opt/boccaccioAI/{fpath}*.jsonl 2>/dev/null | wc -l")
            if out and out != "0K" and out != "4.0K":
                print(f"    {label:25s} {out:>8s}  ({count} shard)")
            else:
                print(f"    {label:25s}   (vuoto)")
        else:
            code, out = run_command(ssh, f"ls -lh /opt/boccaccioAI/{fpath} 2>/dev/null | awk '{{print $5}}'")
            if code == 0 and out:
                print(f"    {label:25s} {out:>8s}")
            else:
                print(f"    {label:25s}   (non creato)")

    print()

    # ─── RAM ──────────────────────────────────────────────
    code, ram_out = run_command(ssh, "free -h | grep Mem | awk '{print $2\" total, \"$3\" used, \"$4\" free\"}'")
    if ram_out:
        print(f"  RAM: {ram_out}")

    print()
    print("=" * 60)

    if stage == "completed":
        print("  FASI 1-2 COMPLETATE!")
        print(f"  Esegui: python scripts/vm_download.py --host {args.host} --password <PASSWORD>")
        print("=" * 60)
    elif not tmux_running and status != "completed":
        print("  ATTENZIONE: la sessione tmux non e' attiva.")
        print("  Il pipeline potrebbe essersi fermato o aver crashato.")
        print("=" * 60)

    ssh.close()


if __name__ == "__main__":
    main()
