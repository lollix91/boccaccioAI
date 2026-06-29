"""BoccaccioAI - Lightning.ai Results Download.

Scarica i checkpoint del training e i log dallo Studio Lightning.ai
sul PC locale.

Uso:
    python scripts/lightning_download.py
    python scripts/lightning_download.py --checkpoints-only

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import os
import sys

import paramiko


# ─── Configurazione Lightning.ai ──────────────────────────────

LIGHTNING_HOST = "ssh.lightning.ai"
LIGHTNING_PORT = 22
LIGHTNING_USER = "s_01kw9jgs29f9znwd4cwpcctbpa"
LIGHTNING_KEY = os.path.expanduser("~/.ssh/lightning_rsa")
PROJECT_DIR = "/home/zeus/content/boccaccioAI"


# ─── Helper ───────────────────────────────────────────────────

def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 30) -> tuple[int, str]:
    """Esegue un comando SSH e ritorna (exit_code, output)."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_code, (out + err).strip()


def download_file(sftp: paramiko.SFTPClient, remote_path: str, local_path: str) -> bool:
    """Scarica un file remoto."""
    try:
        sftp.stat(remote_path)
    except FileNotFoundError:
        print(f"  SKIP: {remote_path} (non trovato)")
        return False
    except IOError:
        print(f"  SKIP: {remote_path} (non accessibile)")
        return False

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    remote_size = sftp.stat(remote_path).st_size

    print(f"  Download: {os.path.basename(remote_path)} ({remote_size / 1e6:.1f} MB)")

    last_print = [0]

    def callback(transferred: int, total: int) -> None:
        if total > 0:
            pct = transferred / total * 100
            if pct - last_print[0] >= 10 or transferred == total:
                last_print[0] = pct
                print(f"    {pct:.0f}%", end="\r", flush=True)

    sftp.get(remote_path, local_path, callback=callback)
    print(f"    100% - completato")
    return True


def download_dir(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: str) -> int:
    """Scarica ricorsivamente una directory."""
    try:
        entries = sftp.listdir_attr(remote_dir)
    except IOError:
        print(f"  SKIP: {remote_dir} (non trovata)")
        return 0

    os.makedirs(local_dir, exist_ok=True)
    count = 0

    for entry in entries:
        remote_path = f"{remote_dir}/{entry.filename}"
        local_path = os.path.join(local_dir, entry.filename)

        try:
            sftp.stat(remote_path)
            if download_file(sftp, remote_path, local_path):
                count += 1
        except IOError:
            count += download_dir(sftp, remote_path, local_path)

    return count


# ─── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BoccaccioAI - Lightning.ai Results Download")
    parser.add_argument("--host", default=LIGHTNING_HOST, help="Studio SSH host")
    parser.add_argument("--port", type=int, default=LIGHTNING_PORT, help="SSH port")
    parser.add_argument("--user", default=LIGHTNING_USER, help="SSH user")
    parser.add_argument("--password", default=None, help="SSH password")
    parser.add_argument("--key", default=LIGHTNING_KEY, help="SSH private key path")
    parser.add_argument("--output-dir", default=".", help="Directory locale di destinazione")
    parser.add_argument("--checkpoints-only", action="store_true", help="Solo checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("  BoccaccioAI - Download Risultati Training")
    print("=" * 60)
    print()

    # ─── Connessione ──────────────────────────────────────────
    print(f"Connessione a {args.host}...", end=" ")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(args.host, port=args.port, username=args.user,
                    password=args.password, key_filename=args.key, timeout=15)
    except Exception as e:
        print(f"FALLITO: {e}")
        sys.exit(1)
    print("OK")

    # ─── Verifica training ────────────────────────────────────
    code, tmux_out = run(ssh, "tmux list-sessions 2>/dev/null")
    tmux_running = "boccaccio" in tmux_out

    if tmux_running:
        print()
        print("ATTENZIONE: il training e' ancora in corso!")
        resp = input("  Scaricare i checkpoint parziali? [y/N] ")
        if resp.lower() != "y":
            print("Download annullato.")
            ssh.close()
            return

    # ─── Lista checkpoint ─────────────────────────────────────
    print()
    print("Checkpoint disponibili:")
    code, ckpt_list = run(ssh, f"ls -lh {PROJECT_DIR}/checkpoints/pretrain/ 2>/dev/null")
    if ckpt_list:
        print(ckpt_list)
    else:
        print("  Nessun checkpoint trovato.")
        ssh.close()
        return
    print()

    # ─── Download ─────────────────────────────────────────────
    sftp = ssh.open_sftp()
    out = args.output_dir

    print("Download file...")
    print()

    downloaded_count = 0

    # Checkpoint
    print("  [Checkpoint pretrain]")
    n = download_dir(sftp, f"{PROJECT_DIR}/checkpoints/pretrain", f"{out}/checkpoints/pretrain")
    downloaded_count += n

    if not args.checkpoints_only:
        # Config
        print()
        print("  [Config]")
        for cfg in ["model.yaml", "training.yaml", "tokenizer.yaml"]:
            download_file(sftp, f"{PROJECT_DIR}/configs/{cfg}", f"{out}/configs/{cfg}")

        # Log
        print()
        print("  [Log]")
        download_file(sftp, f"{PROJECT_DIR}/logs/pretrain_training.log", f"{out}/logs/pretrain_training.log")

        # TensorBoard
        print()
        print("  [TensorBoard logs]")
        n = download_dir(sftp, f"{PROJECT_DIR}/logs/pretrain", f"{out}/logs/pretrain")
        downloaded_count += n

    # Totale
    total_size = 0
    for root, dirs, files in os.walk(f"{out}/checkpoints"):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))

    sftp.close()
    ssh.close()

    print()
    print(f"Download completato: {downloaded_count} elementi, {total_size / 1e6:.1f} MB")
    print()
    print("=" * 60)
    print("  File scaricati in:")
    print(f"    {out}/checkpoints/pretrain/")
    if not args.checkpoints_only:
        print(f"    {out}/configs/")
        print(f"    {out}/logs/")
    print()
    print("  Prossimo passo: Fase 4 (Fine-tuning)")
    print("    bash scripts/04_finetune.sh")
    print()
    print("  Ricordati di FERMARE lo Studio su Lightning.ai!")
    print("=" * 60)


if __name__ == "__main__":
    main()
