"""Launch a command fully detached (its own session, so it survives the
Claude Code process ending — the 22:41 round died with the session; nohup
alone did not protect it on macOS) with caffeinate holding the machine awake.
usage: launch.py LOGFILE -- cmd args..."""
import subprocess
import sys

log = sys.argv[1]
cmd = sys.argv[sys.argv.index("--") + 1:]
with open(log, "w") as f:
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
subprocess.Popen(["caffeinate", "-i", "-w", str(p.pid)], start_new_session=True, close_fds=True,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(p.pid)
