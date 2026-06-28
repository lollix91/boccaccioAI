"""BoccaccioAI - VM Connection & Launch Script.

Si collega alla VM Hetzner via SSH, esegue il setup automatico,
e lancia le Fasi 1-2 in una sessione tmux.

Uso:
    python scripts/vm_connect.py --host <IP> --password <PASSWORD>

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import sys
import time

import paramiko


def run_command(ssh: paramiko.SSHClient, cmd: str, timeout: int = 300) -> tuple[int, str]:
    """Esegue un comando SSH e ritorna (exit_code, output)."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    output = out + err if err else out
    return exit_code, output


def main() -> None:
    parser = argparse.ArgumentParser(description="BoccaccioAI - VM Setup & Launch")
    parser.add_argument("--host", required=True, help="VM IP address")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--user", default="root", help="SSH user")
    parser.add_argument("--password", default=None, help="SSH password")
    parser.add_argument("--repo-url", default="https://github.com/lollix91/boccaccioAI.git",
                        help="Git repo URL")
    parser.add_argument("--skip-setup", action="store_true", help="Skip setup if already done")
    args = parser.parse_args()

    print("=" * 60)
    print("  BoccaccioAI - VM Connection & Launch")
    print("=" * 60)
    print()

    # ─── Connessione SSH ──────────────────────────────────
    print(f"[1/5] Connessione SSH a {args.host}...", end=" ")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(args.host, port=args.port, username=args.user,
                     password=args.password, timeout=30)
    except Exception as e:
        print(f"FALLITO: {e}")
        sys.exit(1)
    print("OK")

    # ─── Verifica sistema ─────────────────────────────────
    print("[2/5] Verifica sistema...", end=" ")
    code, out = run_command(ssh, "uname -a")
    print("OK")
    print(f"       {out.strip()}")

    # ─── Setup ────────────────────────────────────────────
    if not args.skip_setup:
        print("[3/5] Setup automatico (vm_setup.sh)...")
        print("       Questo puo' richiedere 5-10 minuti...")
        print()

        # Trasferisci vm_setup.sh
        sftp = ssh.open_sftp()
        try:
            sftp.put("scripts/vm_setup.sh", "/root/vm_setup.sh")
            sftp.close()
        except FileNotFoundError:
            print("       ERRORE: scripts/vm_setup.sh non trovato in locale.")
            print("       Assicurati di eseguire questo script dalla root del progetto.")
            ssh.close()
            sys.exit(1)

        code, out = run_command(ssh, "bash /root/vm_setup.sh", timeout=600)
        print(out)
        if code != 0:
            print(f"       Setup fallito (exit code {code})")
            ssh.close()
            sys.exit(1)
        print("       Setup completato.")
    else:
        print("[3/5] Setup saltato (--skip-setup)")

    # ─── Crea directory logs ──────────────────────────────
    run_command(ssh, "mkdir -p /opt/boccaccioAI/logs")

    # ─── Installa tmux ────────────────────────────────────
    print("[4/5] Installazione tmux...", end=" ")
    run_command(ssh, "apt-get install -y tmux 2>/dev/null || true")
    print("OK")

    # ─── Lancia pipeline in tmux ──────────────────────────
    print("[5/5] Avvio pipeline in sessione tmux...", end=" ")

    # Uccidi sessione tmux esistente se presente
    run_command(ssh, "tmux kill-session -t boccaccio 2>/dev/null || true")

    # Crea nuova sessione tmux che lancia run_fases_1_2.sh
    tmux_cmd = (
        "tmux new-session -d -s boccaccio "
        "'source /opt/boccaccio-venv/bin/activate && "
        "cd /opt/boccaccioAI && "
        "bash scripts/run_fases_1_2.sh "
        "2>&1 | tee /opt/boccaccioAI/logs/full_pipeline.log'"
    )
    code, out = run_command(ssh, tmux_cmd)
    if code != 0:
        print(f"FALLITO: {out}")
        ssh.close()
        sys.exit(1)

    # Verifica che la sessione sia attiva
    time.sleep(2)
    code, out = run_command(ssh, "tmux list-sessions 2>/dev/null")
    if "boccaccio" not in out:
        print(f"FALLITO: sessione tmux non trovata. Output: {out}")
        ssh.close()
        sys.exit(1)

    print("OK")
    print()
    print("=" * 60)
    print("  Pipeline avviata con successo!")
    print("=" * 60)
    print()
    print("La pipeline sta girando in una sessione tmux sulla VM.")
    print("Puoi spegnere il tuo PC: il processo continua sulla VM.")
    print()
    print("Per monitorare il progresso:")
    print(f"  python scripts/vm_monitor.py --host {args.host} --password <PASSWORD>")
    print()
    print("Per scaricare i risultati al termine:")
    print(f"  python scripts/vm_download.py --host {args.host} --password <PASSWORD>")
    print()
    print("Per collegarti alla sessione tmux interattivamente:")
    print(f"  ssh root@{args.host}")
    print("  tmux attach -t boccaccio")
    print()

    ssh.close()


if __name__ == "__main__":
    main()
