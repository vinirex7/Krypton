# Krypton TradeBot — Aggressive C

A implementação live atual está em `main`. O entrypoint canônico é
`tradebot.py`, que inicia `LiveAggressiveCTradeBot` (`live_c.py`) e utiliza o
motor de decisões de `aggressive_c_live.py`. O repositório inicia em TESTNET;
`USE_TESTNET=false` no ambiente é necessário para produção. O repositório não
informa qual commit está efetivamente instalado em um VPS.

## Configuração live atual

| Regra | Valor |
|---|---|
| Mercado | Binance Spot USDT, somente comprado, sem margem |
| Sleeve tático | 55% do patrimônio |
| Pares/pesos dentro do tático | SOL 31,25%; BTC 50%; BNB 18,75% |
| Risco por entrada tática | 2% da equity tática, limitado por peso e caixa |
| Sinais | Supertrend 7/3, RSI 14 (40–70), MACD 12/26/9 |
| Stop / take profit | 2× / 3× ATR |
| Regime tático | BTC acima da SMA200; gates de continuidade e custo |
| Sleeve alpha | 45%; BTC, ETH, SOL e BNB |
| Seleção alpha | SMA200 própria, momentum 30/90/180 dias, top 2, mínimo 2 |
| Sizing alpha | Inverso da volatilidade/covariância; alvo 30% a.a.; sem alavancagem |
| Rebalanceamento alpha / sleeves | 45 / 90 dias |
| Halt de drawdown | 30% global e tático; persistente |
| Circuit breaker tático | 4% diário |
| Execução | Ciclo 00:05 UTC; candles fechados; LIMIT e OCO tática |

O risco de 2% é sobre o sleeve tático, não sobre o patrimônio inteiro.
`AGGRESSIVE_C_TRANSFER_COST_ASSUMPTION=0.003` é um marcador de pesquisa;
o live Spot atual não debita uma tarifa fictícia de transferência entre sleeves.

## Backtest e validação temporal do live

```bash
pip install -r requirements.txt
python live_validation.py --as-of 2026-10-28 --periods 10m 2y 3y 4y --output results/2026-10-28
```

`--as-of` é o fim exclusivo em UTC. O comando acima inclui até 27/10/2026,
começando respectivamente em 28/12/2025, 28/10/2024, 28/10/2023 e 28/10/2022.
Somente dias UTC completos são aceitos. As quatro moedas têm cobertura diária
integral verificada, mais 400 dias de histórico anterior. Não há fallback para
outro mercado ou proxy USD.

Os entrypoints anteriores agora usam o replay Aggressive C por padrão:

```bash
python backtest.py --start 2024-10-28 --end 2026-10-27 --output results/backtest
python walk_forward.py --start 2024-10-28 --end 2026-10-27 --output results/walk-forward
```

O replay chama o método `AggressiveCTradeBot.daily_cycle` e os mesmos métodos de
sizing, execução e reconciliação do runtime atual. Somente bolsa, relógio,
persistência e telemetria são substituídos. Nenhuma chave, conta Binance ou
ordem real é utilizada. O histórico de cada decisão reproduz a consulta live:
limite 400 com o candle atual aberto, restando 399 candles fechados. Indicadores
não são calculados cumulativamente sobre vários anos quando o live os reinicia
nessa janela móvel.

A validação walk-forward mantém a configuração live fixa e o estado da carteira
entre janelas contíguas de 90 dias, incluindo a janela final parcial. Há 399 candles diários
fechados de histórico causal anterior a cada decisão, sem treinamento de parâmetros ou
seleção de TP. O produto dos retornos das janelas coincide, por construção e
verificação, com o replay contínuo. Isso mede estabilidade temporal da mesma
estratégia; não é um segundo retorno independente nem uma validação intocada:
a configuração foi promovida em agosto/2026, depois de observar parte dos
períodos históricos avaliados.

Arquivos de saída: `summary.json`, `summary.csv`, `manifest.json`, curvas,
ordens executadas, rejeições, exceções do runtime, estado final e folds de cada
período. O manifesto registra configuração/fingerprint, hashes dos dados,
filtros de lote e todas as hipóteses. O patrimônio final é marcado a mercado,
sem venda forçada na última data.

### Limites de execução e problemas reproduzidos

O preço de 00:05 é aproximado pelo OPEN diário. LIMIT aceita é considerada
preenchida integralmente; comissão em USDT segue `FEE_RATE`; preços de MARKET
usam o slippage configurado. Filtros atuais da bolsa não reproduzem eventuais
mudanças históricas. Se SL e TP forem tocados no mesmo candle, prevalece SL.
Candles diários não mostram filas, fills parciais, máximas intradiárias da equity
ou o monitoramento de risco a cada 30 segundos. Portanto, a configuração e as
decisões são compartilhadas, mas fills e resultados não são uma reprodução
exata da conta live.

O replay preserva problemas do runtime, em vez de corrigi-los silenciosamente:
exceções de reconciliação ativam o halt, e sobras de arredondamento podem ser
classificadas como saldo não gerenciado. A reconciliação de OCO também pode
ajustar a baseline de risco pelo valor marcado como se parte do fluxo fosse
externa. Os arquivos `*_runtime_errors.csv` documentam essas ocorrências. Os
módulos live não são alterados por esta correção de pesquisa.

### Pesquisa legada

As funções antigas continuam disponíveis para os diagnósticos existentes.
Para executá-las via CLI, é obrigatório indicar `--legacy`:

```bash
python backtest.py --legacy --symbol SOLUSDT --start 2022-01-01 --end 2026-10-27
python walk_forward.py --legacy --start 2022-01-01 --end 2026-10-27
```

Esses comandos e `deep_validation.py` avaliam o perfil antigo; não devem ser
rotulados como resultados do Aggressive C. A pesquisa C mais recente antes da
promoção estava em `c_only_pre_live_validation.py`, na branch
`research/c-aggressive-prelive-v1`, com dependências em outras branches de
pesquisa. Ela não representa todas as correções operacionais posteriores do live.

## Live

```bash
cp .env.example .env
python tradebot.py
```

Os comandos abaixo enfileiram pedidos para uma instância live já em execução:

```bash
python tradebot.py --decision-now
python tradebot.py --clear-halt
python tradebot.py --rebase-after-manual-change
```

O último comando exige uma alteração manual de saldo confirmada pelo operador;
não deve ser usado para apagar drawdown real. Estado e posições ficam no SQLite.
Para telemetria e dashboard, veja `DASHBOARD.md`.

## Verificação e CI

```bash
python -m unittest discover -s tests -v
```

Push/PR executam testes determinísticos. Pesquisa de mercado é manual no workflow
`Krypton Backtest and Walk-Forward`, que executa `live_validation.py` nos quatro
períodos e publica os artefatos. Um check de testes verde, isoladamente, não
significa que houve execução de backtest com dados de mercado.
