"""BoccaccioAI - Vast.ai Training Monitor.

Si collega all'istanza Vast.ai via SSH e mostra lo stato del training leggendo
le metriche da TensorBoard. Supporta anche il monitoraggio del daemon di
auto-upload checkpoint.

Uso:
    python scripts/vast_monitor.py --host <ip> --port <porta> --key <path_ssh_key>
    python scripts/vast_monitor.py --host 1.2.3.4 --port 12345 --key ~/.ssh/id_rsa --watch

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import paramiko


# ─── Configurazione ───────────────────────────────────────────

PROJECT_DIR = "/workspace/boccaccioAI"
H100_COST_PER_HOUR = 1.933  # Vast.ai H100 SXM
TOTAL_STEPS = 12779


# ─── Helper ───────────────────────────────────────────────────

def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[int, str]:
    """Esegue un comando SSH e ritorna (exit_code, output)."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    combined = (out + err).strip()
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


# ─── TensorBoard reader ───────────────────────────────────────

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

for tag, (fstep, fval, lstep, lval) in results.items():
    print(f"METRIC|{{tag}}|{{fstep}}|{{fval:.6f}}|{{lstep}}|{{lval:.6f}}")
'''


def read_tensorboard_metrics(ssh: paramiko.SSHClient) -> dict:
    """Legge le metriche da TensorBoard sul server."""
    script = TB_SCRIPT.format(project=PROJECT_DIR)
    run(ssh, "mkdir -p /tmp/_tb_check")
    run(ssh, f"cat > /tmp/_tb_check/check.py << 'PYEOF'\n{script}\nPYEOF")
    code, out = run(ssh, "python3 /tmp/_tb_check/check.py 2>&1")
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BoccaccioAI - Vast.ai Training Monitor")
    parser.add_argument("--host", type=str, required=True, help="IP dell'istanza Vast.ai")
    parser.add_argument("--port", type=int, default=22, help="Porta SSH")
    parser.add_argument("--user", type=str, default="root", help="User SSH")
    parser.add_argument("--key", type=str, default=os.path.expanduser("~/.ssh/id_rsa"), help="Chiave SSH privata")
    parser.add_argument("--watch", action="store_true", help="Aggiorna ogni N secondi")
    parser.add_argument("--interval", type=int, default=30, help="Intervallo (sec)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def connect() -> paramiko.SSHClient:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(args.host, port=args.port, username=args.user, key_filename=args.key, timeout=15)
        return ssh

    def run_once(ssh: paramiko.SSHClient) -> None:
        print("=" * 60)
        print("  BoccaccioAI - Monitoraggio Training Vast.ai")
        print("=" * 60)
        print()

        # Stato tmux
        code, tmux_sessions = run(ssh, "tmux list-sessions 2>/dev/null")
        tmux_running = "boccaccio" in tmux_sessions
        uploader_running = "uploader" in tmux_sessions

        # Metriche TensorBoard
        metrics = read_tensorboard_metrics(ssh)

        # GPU
        code, gpu_out = run(ssh, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null")

        # Checkpoint locali
        code, ckpt_out = run(ssh, f"ls -lh {PROJECT_DIR}/checkpoints/pretrain/*.ckpt 2>/dev/null")

        # Tempo processo
        code, pid_out = run(ssh, "pgrep -f 'src.training.train' | head -1 2>/dev/null")
        uptime_str = ""
        if pid_out and pid_out.strip().isdigit():
            pid = pid_out.strip()
            code, uptime = run(ssh, f"ps -o etimes= -p {pid} 2>/dev/null")
            if uptime and uptime.strip().isdigit():
                uptime_str = uptime.strip()

        # Output
        print(f"  Stato training:  {'RUNNING' if tmux_running else 'STOPPED'}")
        print(f"  Stato uploader:  {'RUNNING' if uploader_running else 'STOPPED'}")

        if metrics:
            step = metrics.get("train/loss", {}).get("last_step", 0)
            progress_pct = int(step / TOTAL_STEPS * 100) if TOTAL_STEPS > 0 else 0

            print(f"  Step:       {step} / {TOTAL_STEPS}")
            print(f"  Progresso:  {progress_pct}%")
            print()

            bar_width = 40
            filled = int(bar_width * progress_pct / 100)
            bar = "=" * filled + "-" * (bar_width - filled)
            print(f"  [{bar}] {progress_pct}%")
            print()

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

            if uptime_str:
                elapsed = int(uptime_str)
                print(f"  Tempo trascorso: {format_elapsed(elapsed)}")
                # Per il resume, calcoliamo la velocita' sugli step effettivamente
                # fatti in questa sessione (step corrente - step di partenza).
                # Lo step di partenza lo ricaviamo dal primo step loggato.
                first_step = metrics.get("train/loss", {}).get("first_step", step)
                steps_done = step - first_step if step > first_step else 0
                # Se abbiamo abbastanza step loggati, usa la velocita' reale
                # Altrimenti usa una stima conservativa (8 sec/step su H100 SXM)
                if steps_done > 10 and elapsed > 0:
                    steps_per_sec = steps_done / elapsed
                else:
                    steps_per_sec = 1.0 / 8.0  # ~8 sec/step stima H100 SXM
                remaining = TOTAL_STEPS - step
                if remaining > 0 and steps_per_sec > 0:
                    eta = int(remaining / steps_per_sec)
                    print(f"  Tempo rimanente: {format_elapsed(eta)} (stimato)")
                elapsed_h = elapsed / 3600
                cost = elapsed_h * H100_COST_PER_HOUR
                print(f"  Costo: ~${cost:.2f} ({elapsed_h:.1f}h x ${H100_COST_PER_HOUR}/hr)")
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

        # Checkpoint locali
        if ckpt_out and "No such file" not in ckpt_out:
            print("  Checkpoint locali:")
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

        # Ultimo upload HF
        code, upload_log = run(ssh, f"tail -5 {PROJECT_DIR}/logs/auto_upload.log 2>/dev/null")
        if upload_log and "No such file" not in upload_log:
            print("  Ultimi upload HF Hub:")
            for line in upload_log.split("\n")[-3:]:
                if line.strip():
                    print(f"    {line.strip()}")
            print()

        print("=" * 60)

    # Connessione
    try:
        ssh = connect()
    except Exception as e:
        print(f"ERRORE connessione SSH: {e}")
        sys.exit(1)

    if args.watch:
        try:
            while True:
                ssh = connect()
                run_once(ssh)
                ssh.close()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoraggio interrotto.")
    else:
        run_once(ssh)
        ssh.close()


if __name__ == "__main__":
    main()
