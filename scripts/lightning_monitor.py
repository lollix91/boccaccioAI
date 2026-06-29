"""BoccaccioAI - Lightning.ai Training Monitor.

Si collega allo Studio via SSH e mostra lo stato del training leggendo
le metriche direttamente da TensorBoard (piu' affidabile del parsing
del progress bar di Lightning).

Uso:
    python scripts/lightning_monitor.py
    python scripts/lightning_monitor.py --watch          # aggiorna ogni 30s
    python scripts/lightning_monitor.py --watch --interval 60

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
PROJECT_DIR = "/home/zeus/content/boccaccioAI"
H100_CREDITS_PER_HOUR = 3.5
TOTAL_STEPS = 12779


# ─── Helper ───────────────────────────────────────────────────

def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[int, str]:
    """Esegue un comando SSH e ritorna (exit_code, output)."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    combined = (out + err).strip()
    # Sanitize per Windows console (cp1252)
    return exit_code, combined.encode("ascii", errors="replace").decode("ascii")


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


# ─── TensorBoard reader (eseguito sul server) ─────────────────

TB_SCRIPT = '''
import sys
import os
import glob
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard not installed")
    sys.exit(1)

log_dir = "{project}/logs/pretrain"
versions = sorted(glob.glob(os.path.join(log_dir, "version_*")))
if not versions:
    print("NO_LOGS")
    sys.exit(0)

latest = versions[-1]
ea = EventAccumulator(latest)
ea.Reload()

tags = ea.Tags().get("scalars", [])
if not tags:
    print("NO_TAGS")
    sys.exit(0)

results = {{}}
for tag in tags:
    events = ea.Scalars(tag)
    if events:
        last = events[-1]
        first = events[0]
        results[tag] = (first.step, first.value, last.step, last.value)

# Stampa in formato semplice
for tag, (fstep, fval, lstep, lval) in results.items():
    print(f"METRIC|{{tag}}|{{fstep}}|{{fval:.6f}}|{{lstep}}|{{lval:.6f}}")
'''


def read_tensorboard_metrics(ssh: paramiko.SSHClient) -> dict:
    """Legge le metriche da TensorBoard sul server."""
    script = TB_SCRIPT.format(project=PROJECT_DIR)
    run(ssh, "mkdir -p /tmp/_tb_check")
    run(ssh, f"cat > /tmp/_tb_check/check.py << 'PYEOF'\n{script}\nPYEOF")
    code, out = run(ssh, "/home/zeus/miniconda3/envs/cloudspace/bin/python /tmp/_tb_check/check.py 2>&1")
    run(ssh, "rm -rf /tmp/_tb_check")

    metrics = {}
    for line in out.split("\n"):
        if line.startswith("METRIC|"):
            parts = line.split("|")
            if len(parts) == 6:
                tag = parts[1]
                fstep = int(parts[2])
                fval = float(parts[3])
                lstep = int(parts[4])
                lval = float(parts[5])
                metrics[tag] = {"first_step": fstep, "first_val": fval,
                                "last_step": lstep, "last_val": lval}
    return metrics


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

        # ─── Metriche TensorBoard ────────────────────────────
        metrics = read_tensorboard_metrics(ssh)

        # ─── GPU ─────────────────────────────────────────────
        code, gpu_out = run(ssh, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null")

        # ─── Checkpoint ──────────────────────────────────────
        code, ckpt_out = run(ssh, f"ls -lh {PROJECT_DIR}/checkpoints/pretrain/*.ckpt 2>/dev/null")

        # ─── Tempo processo ──────────────────────────────────
        code, pid_out = run(ssh, "pgrep -f 'src.training.train' | head -1 2>/dev/null")
        uptime_str = ""
        if pid_out and pid_out.strip().isdigit():
            pid = pid_out.strip()
            code, uptime = run(ssh, f"ps -o etimes= -p {pid} 2>/dev/null")
            if uptime and uptime.strip().isdigit():
                uptime_str = uptime.strip()

        # ─── Output ──────────────────────────────────────────
        print(f"  Stato:      {'RUNNING' if tmux_running else 'STOPPED'}")

        if metrics:
            step = metrics.get("train/loss", {}).get("last_step", 0)
            progress_pct = int(step / TOTAL_STEPS * 100) if TOTAL_STEPS > 0 else 0

            print(f"  Step:       {step} / {TOTAL_STEPS}")
            print(f"  Progresso:  {progress_pct}%")
            print()

            # Barra
            bar_width = 40
            filled = int(bar_width * progress_pct / 100)
            bar = "=" * filled + "-" * (bar_width - filled)
            print(f"  [{bar}] {progress_pct}%")
            print()

            # Metriche
            print("  Metriche:")
            if "train/loss" in metrics:
                m = metrics["train/loss"]
                ppl = metrics.get("train/ppl", {}).get("last_val")
                ppl_str = f"  (ppl: {ppl:.2f})" if ppl else ""
                print(f"    Train loss:    {m['last_val']:.4f}{ppl_str}")
            if "val/loss" in metrics:
                m = metrics["val/loss"]
                ppl = metrics.get("val/ppl", {}).get("last_val")
                ppl_str = f"  (ppl: {ppl:.2f})" if ppl else ""
                print(f"    Val loss:      {m['last_val']:.4f}{ppl_str}")
            if "train/lr" in metrics:
                lr = metrics["train/lr"]["last_val"]
                print(f"    Learning rate: {lr:.2e}")
            if "perf/tokens_per_sec" in metrics:
                tps = metrics["perf/tokens_per_sec"]["last_val"]
                print(f"    Tokens/sec:    {tps:,.0f}")
            print()

            # Tempo e costo
            if uptime_str:
                elapsed = int(uptime_str)
                print(f"  Tempo trascorso: {format_elapsed(elapsed)}")
                if step > 0 and elapsed > 0:
                    steps_per_sec = step / elapsed
                    remaining = TOTAL_STEPS - step
                    if remaining > 0 and steps_per_sec > 0:
                        eta = int(remaining / steps_per_sec)
                        print(f"  Tempo rimanente: {format_elapsed(eta)}")
                elapsed_h = elapsed / 3600
                cost = elapsed_h * H100_CREDITS_PER_HOUR
                print(f"  Crediti consumati: ~{cost:.1f} ({elapsed_h:.1f}h x {H100_CREDITS_PER_HOUR})")
                print()
        else:
            print("  (nessuna metrica disponibile ancora)")
            print()

        # GPU
        if gpu_out:
            parts = [p.strip() for p in gpu_out.split(",")]
            if len(parts) >= 4:
                print(f"  GPU:  {parts[0]} util, {parts[1]} / {parts[2]} VRAM, {parts[3]}C")
            print()

        # Checkpoint
        if ckpt_out and "No such file" not in ckpt_out:
            print("  Checkpoint salvati:")
            for line in ckpt_out.split("\n"):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        name = os.path.basename(parts[-1])
                        size = parts[4]
                        print(f"    {name:45s} {size}")
            print()
        else:
            print("  Checkpoint: nessuno ancora salvato")
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
