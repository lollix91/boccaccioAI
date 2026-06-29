"""Check tokenize crash reason."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('167.233.127.36', username='root', password='Ciccio9191', timeout=30)

def run(cmd, timeout=30):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode('utf-8', errors='replace')
    return exit_code, out

# Full tokenize log
code, out = run("cat /opt/boccaccioAI/logs/fase_2c_tokenize.log 2>/dev/null | cat -v | grep -v 'httpx\\|HTTP Request'")
print(f"=== Full fase_2c log ===\n{out}")

# Full pipeline log
code, out = run("cat /opt/boccaccioAI/logs/full_pipeline.log 2>/dev/null | cat -v | grep -v 'httpx\\|HTTP Request' | tail -30")
print(f"\n=== full_pipeline.log (last 30) ===\n{out}")

# Check dmesg for OOM killer
code, out = run("dmesg 2>/dev/null | grep -i 'oom\\|kill\\|memory' | tail -10")
print(f"\n=== dmesg OOM ===\n{out if out.strip() else '(no OOM messages)'}")

# Check tokenize_corpus.py - how does it read data?
code, out = run("head -80 /opt/boccaccioAI/src/data/tokenize_corpus.py | cat -v")
print(f"\n=== tokenize_corpus.py (first 80 lines) ===\n{out}")

ssh.close()
