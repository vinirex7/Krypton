"""Install a per-user launchd job for demo trading using the current interpreter."""
import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    if sys.platform != "darwin":
        sys.exit("This installer requires macOS")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", choices=["binance", "kraken"], default="binance")
    parser.add_argument("--strategy", choices=["KSO1316", "KSL1", "KSL2"], default="KSO1316")
    args = parser.parse_args()
    exchange = args.exchange
    root = Path(__file__).resolve().parent
    credentials = Path.home() / (".krypton-kraken-demo.env" if exchange == "kraken" else ".krypton-demo.env")
    if (not credentials.is_file() or credentials.stat().st_uid != os.getuid()
            or credentials.stat().st_mode & 0o077):
        sys.exit(f"Create a private {credentials} (chmod 600) first")
    label = "com.krypton.kraken.demo" if exchange == "kraken" else "com.krypton.demo"
    target = Path.home() / "Library/LaunchAgents" / (label + ".plist")
    target.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(os.environ.get("KRYPTON_STATE_DIR", root / "var")).resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    command = 'set -ae; . "$1"; exec "$2" "$3" demo --exchange "$4" --strategy "$5"'
    job = dict(Label=label, ProgramArguments=["/bin/sh", "-c", command, label,
               str(credentials), sys.executable, str(root / "krypton.py"), exchange, args.strategy],
               EnvironmentVariables={"KRYPTON_STATE_DIR": str(directory)},
               WorkingDirectory=str(root), KeepAlive=True, ThrottleInterval=30,
               StandardErrorPath=str(directory / ("kraken-demo.log" if exchange == "kraken" else "demo.log")))
    domain = f"gui/{os.getuid()}"
    if subprocess.run(["launchctl", "print", domain + "/" + label],
                      capture_output=True).returncode == 0:
        sys.exit("Demo service already loaded")
    target.write_bytes(plistlib.dumps(job))
    subprocess.run(["launchctl", "bootstrap", domain, str(target)], check=True)
