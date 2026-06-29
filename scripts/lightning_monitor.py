"""BoccaccioAI - Lightning.ai Training Monitor.

Si collega allo Studio via SSH e mostra lo stato del training:
- Step corrente / totale
- Loss e perplexity
- Tempo trascorso e stimato
- GPU utilization e memoria
- Checkpoint salvati
- Crediti consumati

Uso:
    python scripts/lightning_monitor.py
    python scripts/lightning_monitor.py --watch          # aggiorna ogni 30s
    python scripts/lightning_monitor.py --watch --interval 60

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import paramiko


# ─── Configurazione Lightning.ai ──────────────────────────────

LIGHTNING_HOST = "ssh.lightning.ai"
LIGHTNING_PORT = 22
LIGHTNING_USER = "s_01kw9jgs29f9znwd4cwpcctbpa"
LIGHTNING_KEY = os.path.expanduser("~/.ssh/lightning_rsa")
PROJECT_DIR = "/home/zeus/content/boccaccioAI"
H100_CREDITS_PER_HOUR = 3.5


# ─── Helper ───────────────────────────────────────────────────

def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 30) -> tuple[int, str]:
    """Esegue un comando SSH e ritorna (exit_code, output)."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_code, (out + err).strip()


def format_elapsed(seconds: int) -> str:
    """Formatta secondi in h m s."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def parse_training_log(log_text: str) -> dict:
    """Estrae metriche dal log di training di PyTorch Lightning."""
    result = {
        "step": None, "total_steps": None,
        "train_loss": None, "val_loss": None,
        "train_ppl": None, "val_ppl": None,
        "lr": None, "epoch": None, "tokens_per_sec": None,
    }

    lines = log_text.strip().split("\n")

    # Step corrente
    for line in reversed(lines):
        m = re.search(r"global step[:\s]+(\d+)", line, re.IGNORECASE)
        if m:
            result["step"] = int(m.group(1))
            break

    # Total steps (stampato all'inizio)
    for line in lines:
        if "pre-training for" in line.lower() or "total steps" in line.lower():
            m = re.search(r"(\d+)\s+steps?", line, re.IGNORECASE)
            if m:
                result["total_steps"] = int(m.group(1))
                break

    # Train loss
    for line in reversed(lines):
        if "train/loss" in line.lower() or "'train/loss'" in line:
            m = re.search(r"train/loss[:\s]+([\d.]+)", line, re.IGNORECASE)
            if m:
                result["train_loss"] = float(m.group(1))
            m = re.search(r"train/ppl[:\s]+([\d.]+)", line, re.IGNORECASE)
            if m:
                result["train_ppl"] = float(m.group(1))
            break

    # Val loss
    for line in reversed(lines):
        if "val/loss" in line.lower() or "'val/loss'" in line:
            m = re.search(r"val/loss[:\s]+([\d.]+)", line, re.IGNORECASE)
            if m:
                result["val_loss"] = float(m.group(1))
            m = re.search(r"val/ppl[:\s]+([\d.]+)", line, re.IGNORECASE)
            if m:
                result["val_ppl"] = float(m.group(1))
            break

    # LR
    for line in reversed(lines):
        m = re.search(r"lr[:\s]+([\d.e-]+)", line, re.IGNORECASE)
        if m:
            result["lr"] = float(m.group(1))
            break

    # Tokens/sec
    for line in reversed(lines):
        m = re.search(r"tokens.?per.?sec[:\s]+([\d.]+)", line, re.IGNORECASE)
        if m:
            result["tokens_per_sec"] = float(m.group(1))
            break

    # Epoch
    for line in reversed(lines):
        m = re.search(r"epoch[:\s]+(\d+)", line, re.IGNORECASE)
        if m:
            result["epoch"] = int(m.group(1))
            break

    return result


def parse_tmux_progress(tmux_text: str) -> dict:
    """Estrae metriche dallo schermo tmux."""
    result = {"step": None, "train_loss": None, "progress_pct": None}
    lines = tmux_text.strip().split("\n")

    for line in reversed(lines):
        m = re.search(r"(\d+)%.*?(\d+)/(\d+)", line)
        if m:
            result["progress_pct"] = int(m.group(1))
            result["step"] = int(m.group(2))
            break

    for line in reversed(lines):
        m = re.search(r"loss[=: ]+([\d.]+)", line, re.IGNORECASE)
        if m:
            result["train_loss"] = float(m.group(1))
            break

    return result


# ─── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BoccaccioAI - Lightning.ai Training Monitor")
    parser.add_argument("--host", default=LIGHTNING_HOST, help="Studio SSH host")
    parser.add_argument("--port", type=int, default=LIGHTNING_PORT, help="SSH port")
    parser.add_argument("--user", default=LIGHTNING_USER, help="SSH user")
    parser.add_argument("--password", default=None, help="SSH password")
    parser.add_argument("--key", default=LIGHTNING_KEY, help="SSH private key path")
    parser.add_argument("--watch", action="store_true", help="Aggiorna ogni N secondi")
    parser.add_argument("--interval", type=int, default=30, help="Intervallo (sec)")
    args = parser.parse_args()

    def connect() -> paramiko.SSHClient:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(args.host, port=args.port, username=args.user,
                    password=args.password, key_filename=args.key, timeout=15)
        return ssh

    def run_once(ssh: paramiko.SSHClient) -> None:
        print("=" * 60)
        print("  BoccaccioAI - Monitoraggio Training Lightning.ai")
        print("=" * 60)
        print()

        # ─── Stato tmux ──────────────────────────────────────
        code, tmux_sessions = run(ssh, "tmux list-sessions 2>/dev/null")
        tmux_running = "boccaccio" in tmux_sessions

        # ─── Schermo tmux ────────────────────────────────────
        tmux_text = ""
        if tmux_running:
            code, tmux_text = run(ssh, "tmux capture-pane -t boccaccio -p -S -50 2>/dev/null")

        # ─── Log file ────────────────────────────────────────
        code, log_text = run(ssh, f"tail -200 {PROJECT_DIR}/logs/pretrain_training.log 2>/dev/null")

        # ─── Parsing ─────────────────────────────────────────
        log_data = parse_training_log(log_text) if log_text else {}
        tmux_data = parse_tmux_progress(tmux_text) if tmux_text else {}

        step = tmux_data.get("step") or log_data.get("step")
        total_steps = log_data.get("total_steps")
        train_loss = log_data.get("train_loss") or tmux_data.get("train_loss")
        val_loss = log_data.get("val_loss")
        train_ppl = log_data.get("train_ppl")
        val_ppl = log_data.get("val_ppl")
        lr = log_data.get("lr")
        tokens_per_sec = log_data.get("tokens_per_sec")
        epoch = log_data.get("epoch")
        progress_pct = tmux_data.get("progress_pct")

        if step and total_steps and not progress_pct:
            progress_pct = int(step / total_steps * 100)

        # ─── Stato ───────────────────────────────────────────
        print(f"  Stato:      {'RUNNING' if tmux_running else 'STOPPED'}")
        if step is not None:
            print(f"  Step:       {step}" + (f" / {total_steps}" if total_steps else ""))
        if progress_pct is not None:
            print(f"  Progresso:  {progress_pct}%")
        if epoch is not None:
            print(f"  Epoch:      {epoch}")
        print()

        # ─── Barra ───────────────────────────────────────────
        if progress_pct is not None and progress_pct >= 0:
            bar_width = 40
            filled = int(bar_width * progress_pct / 100)
            bar = "=" * filled + "-" * (bar_width - filled)
            print(f"  [{bar}] {progress_pct}%")
            print()

        # ─── Metriche ────────────────────────────────────────
        print("  Metriche:")
        if train_loss is not None:
            ppl_str = f"  (ppl: {train_ppl:.2f})" if train_ppl else ""
            print(f"    Train loss:    {train_loss:.4f}{ppl_str}")
        if val_loss is not None:
            ppl_str = f"  (ppl: {val_ppl:.2f})" if val_ppl else ""
            print(f"    Val loss:      {val_loss:.4f}{ppl_str}")
        if lr is not None:
            print(f"    Learning rate: {lr:.2e}")
        if tokens_per_sec is not None:
            print(f"    Tokens/sec:    {tokens_per_sec:.0f}")
        print()

        # ─── GPU ─────────────────────────────────────────────
        code, gpu_out = run(ssh, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null")
        if gpu_out:
            parts = [p.strip() for p in gpu_out.split(",")]
            if len(parts) >= 4:
                print(f"  GPU:  {parts[0]} util, {parts[1]} / {parts[2]} VRAM, {parts[3]}C")
            print()

        # ─── Checkpoint ──────────────────────────────────────
        code, ckpt_out = run(ssh, f"ls -lh {PROJECT_DIR}/checkpoints/pretrain/*.ckpt 2>/dev/null")
        if ckpt_out:
            print("  Checkpoint salvati:")
            for line in ckpt_out.split("\n"):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        name = os.path.basename(parts[-1])
                        size = parts[4]
                        print(f"    {name:40s} {size}")
            print()
        else:
            print("  Checkpoint: nessuno ancora salvato")
            print()

        # ─── Schermo tmux ────────────────────────────────────
        if tmux_running and tmux_text:
            lines = [l for l in tmux_text.split("\n") if l.strip()]
            useful = [l for l in lines if "httpx" not in l and "HTTP Request" not in l]

            print(f"  Schermo tmux (ultime {min(15, len(useful))} righe):")
            print("  " + "-" * 56)
            for line in useful[-15:]:
                clean = line.replace("\r", "").strip()
                if len(clean) > 120:
                    clean = clean[:120] + "..."
                print(f"  {clean}")
            print()

        # ─── Tempo e costo ───────────────────────────────────
        code, uptime = run(ssh, "ps -o etimes= -p $(pgrep -f 'src.training.train' | head -1) 2>/dev/null")
        if uptime and uptime.strip().isdigit():
            elapsed = int(uptime.strip())
            print(f"  Tempo trascorso: {format_elapsed(elapsed)}")
            if step and total_steps and step > 0 and elapsed > 0:
                steps_per_sec = step / elapsed
                remaining = total_steps - step
                if remaining > 0 and steps_per_sec > 0:
                    eta = int(remaining / steps_per_sec)
                    print(f"  Tempo rimanente: {format_elapsed(eta)}")
            elapsed_h = elapsed / 3600
            cost = elapsed_h * H100_CREDITS_PER_HOUR
            print(f"  Crediti consumati: ~{cost:.1f} ({elapsed_h:.1f}h x {H100_CREDITS_PER_HOUR})")
            print()

        print("=" * 60)

    # ─── Loop ────────────────────────────────────────────────
    if args.watch:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                try:
                    ssh = connect()
                    run_once(ssh)
                    ssh.close()
                except Exception as e:
                    print(f"  Errore connessione: {e}")
                print(f"\n  Prossimo aggiornamento tra {args.interval}s (Ctrl+C per uscire)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\n  Monitoraggio fermato.")
    else:
        try:
            ssh = connect()
            run_once(ssh)
            ssh.close()
        except Exception as e:
            print(f"ERRORE: Impossibile connettersi: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
