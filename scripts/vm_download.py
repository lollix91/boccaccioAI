"""BoccaccioAI - VM Results Download.

Si collega alla VM via SSH e scarica tutti i file generati
dalle Fasi 1-2 (tokenizer + dati pre-tokenizzati).

Uso:
    python scripts/vm_download.py --host <IP> --password <PASSWORD>

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import paramiko


def download_file(sftp: paramiko.SFTPClient, remote_path: str, local_path: str) -> bool:
    """Scarica un file remoto. Ritorna True se successo, False se file non esiste."""
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

    print(f"  Download: {remote_path} ({remote_size / 1e6:.1f} MB) -> {local_path}")

    # Download con progress callback
    downloaded = [0]
    last_print = [0]

    def callback(transferred, total):
        downloaded[0] = transferred
        if total > 0:
            pct = transferred / total * 100
            if pct - last_print[0] >= 10 or transferred == total:
                last_print[0] = pct
                print(f"    {pct:.0f}%", end="\r", flush=True)

    sftp.get(remote_path, local_path, callback=callback)
    print(f"    100% - completato")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="BoccaccioAI - VM Results Download")
    parser.add_argument("--host", required=True, help="VM IP address")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--user", default="root", help="SSH user")
    parser.add_argument("--password", default=None, help="SSH password")
    parser.add_argument("--output-dir", default=".", help="Directory locale di destinazione")
    args = parser.parse_args()

    print("=" * 60)
    print("  BoccaccioAI - Download Risultati VM")
    print("=" * 60)
    print()

    # ─── Connessione ──────────────────────────────────────
    print(f"Connessione a {args.host}...", end=" ")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(args.host, port=args.port, username=args.user,
                     password=args.password, timeout=15)
    except Exception as e:
        print(f"FALLITO: {e}")
        sys.exit(1)
    print("OK")

    # ─── Verifica completamento ───────────────────────────
    # Check if train.bin exists (pipeline completed)
    stdin, stdout, stderr = ssh.exec_command(
        "ls /opt/boccaccioAI/data/tokenized/pretrain/train.bin 2>/dev/null"
    )
    train_bin_check = stdout.read().decode("utf-8", errors="replace").strip()

    if not train_bin_check:
        print()
        print("ATTENZIONE: train.bin non trovato - il pipeline potrebbe non essere completato.")
        resp = input("  Continuare comunque con il download? [y/N] ")
        if resp.lower() != "y":
            print("Download annullato.")
            ssh.close()
            return

    # ─── Download file ────────────────────────────────────
    print()
    print("Download file...")
    print()

    sftp = ssh.open_sftp()
    base = "/opt/boccaccioAI"
    out = args.output_dir

    files = [
        (f"{base}/tokenizer/boccaccio-32k.json", f"{out}/tokenizer/boccaccio-32k.json"),
        (f"{base}/data/tokenized/pretrain/train.bin", f"{out}/data/tokenized/pretrain/train.bin"),
        (f"{base}/data/tokenized/pretrain/val.bin", f"{out}/data/tokenized/pretrain/val.bin"),
        (f"{base}/data/tokenized/pretrain/meta.json", f"{out}/data/tokenized/pretrain/meta.json"),
    ]

    downloaded_count = 0
    total_size = 0

    for remote, local in files:
        if download_file(sftp, remote, local):
            downloaded_count += 1
            total_size += os.path.getsize(local)

    sftp.close()
    ssh.close()

    print()
    print(f"Download completato: {downloaded_count} file, {total_size / 1e6:.1f} MB totali")
    print()
    print("=" * 60)
    print("  File scaricati in:")
    print(f"    {out}/tokenizer/boccaccio-32k.json")
    print(f"    {out}/data/tokenized/pretrain/train.bin")
    print(f"    {out}/data/tokenized/pretrain/val.bin")
    print(f"    {out}/data/tokenized/pretrain/meta.json")
    print()
    print("  Prossimo passo: Fase 2.5 (smoke test su GPU locale)")
    print("    bash scripts/02_5_smoke_test.sh")
    print()
    print("  Ricordati di ELIMINARE la VM dalla console Hetzner!")
    print("=" * 60)


if __name__ == "__main__":
    main()
