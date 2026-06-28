"""Fix run_fases_1_2.sh line endings and launch pipeline in tmux."""
import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('167.233.127.36', username='root', password='Ciccio9191', timeout=30)

# Fix CRLF on run_fases_1_2.sh
stdin, stdout, stderr = ssh.exec_command("sed -i 's/\r//g' /opt/boccaccioAI/scripts/run_fases_1_2.sh", timeout=30)
print("Fix CRLF run_fases_1_2.sh:", stdout.channel.recv_exit_status())

# Create logs directory
stdin, stdout, stderr = ssh.exec_command("mkdir -p /opt/boccaccioAI/logs", timeout=30)
stdout.channel.recv_exit_status()

# Install tmux
stdin, stdout, stderr = ssh.exec_command("apt-get install -y tmux 2>/dev/null", timeout=60)
stdout.channel.recv_exit_status()

# Kill existing tmux session
stdin, stdout, stderr = ssh.exec_command("tmux kill-session -t boccaccio 2>/dev/null || true", timeout=10)
stdout.channel.recv_exit_status()

# Launch pipeline in tmux
tmux_cmd = (
    "tmux new-session -d -s boccaccio "
    "'source /opt/boccaccio-venv/bin/activate && "
    "cd /opt/boccaccioAI && "
    "bash scripts/run_fases_1_2.sh "
    "2>&1 | tee /opt/boccaccioAI/logs/full_pipeline.log'"
)
stdin, stdout, stderr = ssh.exec_command(tmux_cmd, timeout=30)
exit_code = stdout.channel.recv_exit_status()
print(f"Launch tmux: exit {exit_code}")

# Verify session is running
time.sleep(3)
stdin, stdout, stderr = ssh.exec_command("tmux list-sessions 2>/dev/null", timeout=10)
out = stdout.read().decode('utf-8', errors='replace')
print(f"tmux sessions: {out.strip()}")

# Check if progress.json exists yet
stdin, stdout, stderr = ssh.exec_command("cat /opt/boccaccioAI/progress.json 2>/dev/null", timeout=10)
out = stdout.read().decode('utf-8', errors='replace')
print(f"progress.json: {out.strip() if out.strip() else '(not yet created)'}")

# Check last lines of log
stdin, stdout, stderr = ssh.exec_command("tail -5 /opt/boccaccioAI/logs/full_pipeline.log 2>/dev/null", timeout=10)
out = stdout.read().decode('utf-8', errors='replace')
print(f"Last log lines:\n{out}")

ssh.close()
print("\nDone. Pipeline is running in tmux session 'boccaccio'.")
