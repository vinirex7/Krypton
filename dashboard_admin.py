"""Administrative CLI to create/update Krypton dashboard users."""
from __future__ import annotations

import argparse
import getpass
import os

from telemetry import TelemetryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision Krypton dashboard client")
    parser.add_argument("--id", required=True, dest="client_id")
    parser.add_argument("--name", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", default=None)
    args = parser.parse_args()

    password = args.password or getpass.getpass("Password (min. 10 chars): ")
    store = TelemetryStore(os.getenv("KRYPTON_DASHBOARD_DB", "krypton_dashboard.db"))
    store.ensure_client(args.client_id, args.name, args.email, password)
    print(f"Client {args.client_id!r} provisioned for {args.email}")


if __name__ == "__main__":
    main()
