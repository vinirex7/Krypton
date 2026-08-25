# 🔷 Krypton TradeBot

Tradebot spot para Binance com **Supertrend + RSI + MACD**, sizing por ATR e controles de risco. O repositório agora mantém o backtest e o live alinhados com a execução real de spot.

## Moedas de cotação

- **Backtest:** `USDT` (`SOLUSDT`, `BTCUSDT`, `ETHUSDT`, `BNBUSDT`).
- **Live:** `U` (`SOLU`, `BTCU`, `ETHU`, `BNBU`).
- `U` e `USDT` são tratados pelo sistema como a mesma quote asset operacional; a diferença é somente o símbolo usado em cada ambiente.

## Correções de backtest

1. SL/TP usam **LOW/HIGH** do candle; se ambos forem tocados, **SL tem prioridade**.
2. Sinal do CLOSE de `i` só entra no **OPEN de `i+1`** — sem look-ahead.
3. Backtest é **spot LONG-only**; sinal SHORT nunca vira posição short.
4. Sizing usa **1% do capital total**, sem multiplicar pelo percentual de alocação do par.
5. Os pares backtestáveis são os mesmos ativos operados no live, com quote `USDT`.
6. `STOP_LOSS_ATR_MULT`, `TAKE_PROFIT_ATR_MULT`, `CIRCUIT_BREAKER_PCT`, `MAX_DRAWDOWN_PCT` e `MAX_SIMULTANEOUS_POS` vêm do config.
7. ATR é calculado uma única vez sobre o dataset completo com warm-up de 300 dias.
8. Existe cap explícito de notional para impedir alavancagem implícita no spot.
9. Supertrend não propaga `direction=0` após o warm-up.

## Correções de execução live

- Uma LIMIT aceita pela Binance **não é registrada como posição até `FILLED`**.
- Timeout cancela a ordem e reconcilia `get_order`/`get_open_orders`, inclusive fills parciais.
- Entradas preenchidas recebem **OCO real na Binance** para SL/TP.
- O estado das posições é persistido em `krypton_state.db` e reconciliado com o saldo real no boot.
- Se o VPS reiniciar, o bot tenta restaurar a proteção OCO das posições existentes.
- `NOTIONAL` e `MIN_NOTIONAL` são aceitos ao ler os filtros do símbolo.
- A validação de slippage compara o preço da ordem com o **preço do sinal**, não com um mid recém-consultado.
- Não há shorts: o live opera somente spot.
- O ciclo diário é calculado explicitamente em **UTC**, sem depender do timezone local do VPS ou do `schedule`.
- `datetime.now(timezone.utc)` substitui `datetime.utcnow()`.

## Gestão de risco

| Parâmetro | Valor |
|---|---:|
| Risco por trade | 1% do capital total |
| Stop Loss | 2× ATR |
| Take Profit | 4,5× ATR |
| R:R | 2,25:1 |
| Win rate de equilíbrio antes de custos | ~30,8% |
| Circuit breaker | -4% no dia |
| Max drawdown | -20% |
| Máx. posições | 4 |
| Ordem | LIMIT + OCO |

O TP foi elevado de 3× para **4,5× ATR** como ponto de partida para a nova avaliação. Isso não deve ser interpretado como garantia de edge: os parâmetros precisam ser avaliados em **walk-forward/out-of-sample**, e não escolhidos apenas pelo melhor resultado no período inteiro.

## Uso

### Backtest

```bash
python backtest.py --symbol SOLUSDT --start 2022-01-01 --end 2026-06-01
python backtest.py --symbol BTCUSDT --start 2022-01-01
python backtest.py --symbol ETHUSDT --start 2022-01-01
python backtest.py --symbol BNBUSDT --start 2022-01-01
```

### Live/testnet

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python tradebot.py
```

`USE_TESTNET = True` continua como padrão. Só altere para produção depois de validar a execução, fills, OCO, reconciliação e comportamento do risco em testnet.

## Estrutura

```text
Krypton/
├── config.py
├── indicators.py
├── risk_manager.py
├── binance_client.py
├── tradebot.py
├── backtest.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Segurança

- Nunca habilite saque na API key.
- Use somente as permissões necessárias de leitura/trading spot.
- Restrinja a API ao IP do VPS quando possível.
- Não use capital que você não possa perder.
- Teste primeiro em testnet.
