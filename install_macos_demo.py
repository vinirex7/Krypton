"""Install a per-user launchd job for demo trading using the current interpreter."""
import os
import plistlib
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    if sys.platform != "darwin":
        sys.exit("This installer requires macOS")
    root = Path(__file__).resolve().parent
    credentials = Path.home() / ".krypton-demo.env"
    if (not credentials.is_file() or credentials.stat().st_uid != os.getuid()
            or credentials.stat().st_mode & 0o077):
        sys.exit("Create a private ~/.krypton-demo.env (chmod 600) first")
    label = "com.krypton.demo"
    target = Path.home() / "Library/LaunchAgents" / (label + ".plist")
    target.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(os.environ.get("KRYPTON_STATE_DIR", root / "var")).resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    command = 'set -ae; . "$1"; exec "$2" "$3" demo'
    job = dict(Label=label, ProgramArguments=["/bin/sh", "-c", command, label,
               str(credentials), sys.executable, str(root / "krypton.py")],
               EnvironmentVariables={"KRYPTON_STATE_DIR": str(directory)},
               WorkingDirectory=str(root), KeepAlive=True, ThrottleInterval=30,
               StandardErrorPath=str(directory / "demo.log"))
    domain = f"gui/{os.getuid()}"
    if subprocess.run(["launchctl", "print", domain + "/" + label],
                      capture_output=True).returncode == 0:
        sys.exit("Demo service already loaded")
    target.write_bytes(plistlib.dumps(job))
    subprocess.run(["launchctl", "bootstrap", domain, str(target)], check=True)
