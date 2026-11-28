"""Initial BRL allocation for the dedicated Kraken KSL2 demo wallet."""
import json
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from urllib.request import urlopen

FX_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados/ultimos/1?formato=json"


def initial_budget():
    with urlopen(FX_URL, timeout=8) as response:
        row = json.load(response)[-1]
    rate = Decimal(str(row["valor"]))
    day = datetime.strptime(row["data"], "%d/%m/%Y").date()
    age = (datetime.now(timezone.utc).date() - day).days
    if not rate.is_finite() or rate <= 0 or not 0 <= age <= 7:
        raise ValueError("Cotação USD/BRL do BCB inválida ou desatualizada")
    usd = (Decimal(100) / rate).quantize(Decimal(".01"), rounding=ROUND_DOWN)
    if usd <= 0:
        raise ValueError("Orçamento demo inválido")
    return usd, rate, day.isoformat()


def configure_budget(api, state):
    if "demo_100_brl" in state.data:
        return
    if state.data["trade"] or state.data["seen"] or Decimal(state.data["peak"]):
        raise RuntimeError("Use um estado novo e uma carteira demo dedicada e livre para o teste de R$ 100")
    if api.position()[0] or api.open_orders():
        raise RuntimeError("A carteira demo precisa iniciar sem posições ou ordens")
    usd, rate, day = initial_budget()
    wallet, free = api.balances()
    if min(wallet, free) < usd:
        raise RuntimeError("Saldo disponível na demo abaixo do equivalente de R$ 100")
    state.data["demo_100_brl"] = {"initial_usd": str(usd), "brl_per_usd": str(rate),
                                  "fx_date": day, "reserve_usd": str(wallet-usd)}
    state.data["peak"] = str(usd)
    state.save()
