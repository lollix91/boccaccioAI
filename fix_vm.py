"""Check VM logs to diagnose download failure."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('167.233.127.36', username='root', password='Ciccio9191', timeout=30)

def run(cmd, timeout=30):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    return exit_code, out, err

# Check download log
print("=== Fase 2a download log (last 30 lines) ===")
code, out, _ = run("tail -30 /opt/boccaccioAI/logs/fase_2a_download.log 2>/dev/null | cat -v")
print(out)

# Check if data/raw has anything
print("\n=== data/raw contents ===")
code, out, _ = run("ls -la /opt/boccaccioAI/data/raw/ 2>/dev/null")
print(out if out.strip() else "(empty or doesn't exist)")

# Check full pipeline log (last 50 lines)
print("\n=== Full pipeline log (last 50 lines) ===")
code, out, _ = run("tail -50 /opt/boccaccioAI/logs/full_pipeline.log 2>/dev/null | cat -v")
print(out)

# Check if HF_TOKEN is set in the venv activate
print("\n=== HF_TOKEN in venv activate ===")
code, out, _ = run("grep HF_TOKEN /opt/boccaccio-venv/bin/activate 2>/dev/null")
print(out if out.strip() else "(not found)")

ssh.close()
