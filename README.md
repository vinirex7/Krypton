# 🔷 Krypton TradeBot

Tradebot spot para Binance com **Supertrend + RSI + MACD**, sizing por ATR, filtro de regime BTC/SMA200 e controles de risco persistentes. Live e pesquisa compartilham a mesma arquitetura congelada.

## Moedas de cotação

- **Backtest e live:** `USDT` (`SOLUSDT`, `BTCUSDT`, `BNBUSDT`).
- Proxies USD/Yahoo não são usados por padrão e nunca são rotulados como USDT.

## Correções de backtest

1. SL/TP usam **LOW/HIGH** do candle; se ambos forem tocados, **SL tem prioridade**.
2. Sinal do CLOSE de `i` só entra no **OPEN de `i+1`** — sem look-ahead.
3. Backtest é **spot LONG-only**; sinal SHORT nunca vira posição short.
4. Sizing arrisca **1% da equity**, limitado pelo peso do ativo e pelo caixa livre no portfólio.
5. Os pares e pesos do walk-forward são os mesmos do live, com quote `USDT`; ETH foi removido da arquitetura validada.
6. `STOP_LOSS_ATR_MULT`, `TAKE_PROFIT_ATR_MULT`, `CIRCUIT_BREAKER_PCT`, `MAX_DRAWDOWN_PCT` e `MAX_SIMULTANEOUS_POS` vêm do config.
7. ATR é calculado uma única vez sobre o dataset completo com warm-up de 300 dias.
8. Existe cap explícito de notional por peso e caixa livre para impedir alavancagem implícita no spot.
9. Supertrend não propaga `direction=0` após o warm-up.
10. Saídas por sinal acontecem no open antes de SL/TP intradiários; gaps e slippage adverso são modelados.
11. Equity, Sharpe, drawdown e circuit breaker usam marcação a mercado diária.
12. Monte Carlo usa block bootstrap dos retornos diários da carteira, não retorno integral de cada posição.

## Correções de execução live

- Uma LIMIT aceita pela Binance **não é registrada como posição até `FILLED`**.
- O sinal diário usa somente candles cujo `close_time` já passou; o candle aberto das 00:05 UTC é descartado.
- Timeout cancela a ordem e reconcilia `get_order`/`get_open_orders`, inclusive fills parciais.
- Entradas preenchidas recebem **OCO real na Binance** para SL/TP.
- A perna de stop da OCO é `STOP_LOSS` a mercado, evitando uma ordem stop-limit presa após um gap.
- O estado das posições é persistido em `krypton_state.db` e reconciliado com o saldo real no boot.
- Se o VPS reiniciar, o bot tenta restaurar a proteção OCO das posições existentes.
- `NOTIONAL` e `MIN_NOTIONAL` são aceitos ao ler os filtros do símbolo.
- A validação de slippage compara o preço da ordem com o **preço do sinal**, não com um mid recém-consultado.
- Não há shorts: o live opera somente spot.
- Novas entradas exigem BTC acima da SMA200; saídas continuam permitidas em risk-off.
- Estado de peak equity, drawdown halt e circuit breaker é persistido no SQLite.
- Falta da OCO específica da posição bloqueia novas entradas.
- Saldos manuais não são adotados como posições do bot por padrão.
- Vendas, retiradas ou reduções simultâneas de cripto e USDT são reconciliadas
  pelo fluxo externo líquido, sem transformar retirada de capital em drawdown.
- O ciclo diário é calculado explicitamente em **UTC**, sem depender do timezone local do VPS ou do `schedule`.
- `datetime.now(timezone.utc)` substitui `datetime.utcnow()`.
- Um boot fora da janela das 00:05–00:09 UTC não envia entrada atrasada no meio do candle.

## Gestão de risco

| Parâmetro | Valor |
|---|---:|
| Risco por trade | 1% do capital total |
| Stop Loss | 2× ATR |
| Take Profit | 3× ATR |
| R:R | 1,5:1 |
| Filtro de regime | BTC > SMA200 |
| Circuit breaker | -4% no dia |
| Max drawdown | -20% |
| Máx. posições | 3 |
| Ordem | LIMIT + OCO |

O TP live está congelado em **3× ATR**, conforme a arquitetura levada ao holdout final. Qualquer alteração exige um novo processo de seleção pré-holdout e validação OOS.

## Uso

### Backtest

```bash
python backtest.py --symbol SOLUSDT --start 2022-01-01 --end 2026-06-01
python backtest.py --symbol BTCUSDT --start 2022-01-01
python backtest.py --symbol BNBUSDT --start 2022-01-01
python walk_forward.py --start 2022-01-01
python deep_validation.py --start 2022-01-01
```

### Diagnóstico profundo

`deep_validation.py` mantém a seleção de TP restrita ao período anterior a cada holdout e também gera:

- auditoria de cobertura, datas ausentes, duplicatas, `NaN` e OHLC inválido por ativo;
- funil diário de sinais com o motivo exato de cada bloqueio;
- `diagnostics_2026.csv` para investigar períodos sem operações;
- atribuição de PnL por ativo, ano, holdout e motivo de saída;
- stress com 0, 5, 10, 20 e 50 bps adicionais de slippage por lado;
- estabilidade diagnóstica para TP entre 2,5 e 5,0 ATR, sem usar o holdout para selecionar o TP;
- curva OOS cronológica, incluindo retorno zero nos intervalos entre holdouts.

Execução completa, salvando o relatório de terminal:

```bash
python deep_validation.py --start 2022-01-01 --diagnostics-year 2026 \
  --stress-bps 0 5 10 20 50 --tp-grid 2.5 3.0 3.5 4.0 4.5 5.0 \
  2>&1 | tee deep_validation_resultado.txt
```

Arquivos centrais: `deep_validation_data_audit.csv`, `deep_validation_signal_funnel.csv`,
`deep_validation_signal_diagnostics.csv`, `diagnostics_2026.csv`,
`deep_validation_attribution.csv`, `deep_validation_cost_stress.csv`,
`deep_validation_tp_stability.csv` e `deep_validation_continuous_oos.csv`.

Por integridade, falta de acesso ao mercado Binance USDT encerra a pesquisa com erro; o sistema não troca silenciosamente para USD/Yahoo. No GitHub Actions, os testes determinísticos rodam a cada push e a pesquisa de mercado é iniciada manualmente (`workflow_dispatch`) em um ambiente com acesso à Binance.

### Live/testnet

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python tradebot.py
```

Se uma versão anterior já salvou um halt falso após uma alteração manual de
saldo confirmada, solicite uma única reparação da baseline e então inicie o bot:

```bash
python tradebot.py --rebase-after-manual-change
python tradebot.py
```

O comando preserva posições, ordens e o banco; ele apenas redefine as bases de
risco para o patrimônio Spot já reconciliado. Não use para apagar drawdown real.

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
