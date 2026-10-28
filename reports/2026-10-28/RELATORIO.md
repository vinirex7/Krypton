# Auditoria Krypton — 28/10/2026

## Resultado principal

A `main` era a implementação live mais recente no repositório, no commit `cedce58fc59d330dd2092b813f88ec360da3fa43` (09/09/2026 UTC). O caminho é `tradebot.py` → `live_c.LiveAggressiveCTradeBot` → `aggressive_c_live.AggressiveCTradeBot`. Não houve acesso ao VPS nem à conta Binance; não é possível confirmar o commit ou os overrides realmente instalados no servidor.

O backtest e o walk-forward antigos divergiam do live. A correção adiciona `live_validation.py`, reutiliza o método de decisões e a reconciliação do runtime, atualiza os dois entrypoints e o workflow e mantém a execução legada apenas com `--legacy`. Os arquivos live/config/indicadores/risco não foram modificados.

## Diferenças corrigidas

| Item | Pesquisa antiga | Replay atual |
|---|---|---|
| Composição | Tático isolado | 55% tático + 45% alpha |
| Risco tático por entrada | 1% | 2% da equity tática |
| Alpha | Ausente | BTC/ETH/SOL/BNB, top 2, breadth mínimo 2, alvo vol. 30% |
| Limite DD | 20% | 30% global e tático, como o live |
| Rebalanceamentos | Sem dois sleeves | Alpha 45 dias; sleeves 90 dias |
| Gates de continuidade/custo | Ausentes | Funções do live |
| Indicadores | Histórico acumulado | Janela móvel live de 399 candles fechados |
| Sizing/fills | Aproximação legada | Métodos live com bolsa simulada, lote/tick e gate de preço do sinal |
| TP no walk-forward | Seleção em grade | TP fixo 3 ATR, configuração atual |
| Estado entre folds | Reiniciado | Contínuo, inclusive halts |
| Falhas operacionais | Não reproduzidas | Exceções e halt do runtime preservados |

A pesquisa Aggressive C pré-live mais recente estava em `c_only_pre_live_validation.py`, na branch `research/c-aggressive-prelive-v1`, apoiada em `aggressive_risk_frontier_validation.py` e dependências nas branches de pesquisa. Ela foi preservada e não inclui todas as correções operacionais de setembro.

## Execuções de mercado

Dados oficiais Binance Spot USDT, até 27/10/2026 inclusive, com 1.861 candles por ativo incluindo o histórico anterior. Capital inicial independente de **10.000 USDT em caixa** em cada período; custos incluídos; patrimônio final marcado a mercado, sem liquidação forçada. Dois replays completos produziram os mesmos retornos e capitais finais.

| Período | Início | Retorno líquido | CAGR | DD máximo diário | Sharpe | Capital final USDT | Folds WF | Primeiro halt operacional |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 10 meses | 2025-12-28 | +1.04% | 1.25% | -7.85% | 0.192 | 10104.06 | 4 | Nenhum |
| 2 anos | 2024-10-28 | +19.99% | 9.55% | -5.44% | 1.015 | 11998.64 | 9 | 2024-12-08 |
| 3 anos | 2023-10-28 | +50.32% | 14.55% | -20.33% | 0.943 | 15032.16 | 13 | 2024-10-05 |
| 4 anos | 2022-10-28 | +66.44% | 13.58% | -11.76% | 1.010 | 16643.90 | 17 | 2024-04-08 |

**As janelas de 2, 3 e 4 anos foram afetadas por bloqueios operacionais.** Não representam uma estratégia que operou normalmente por todo o horizonte. O DD é da equity diária efetiva, não o DD interno ajustado pelo live. Sharpe e CAGR dessas janelas incluem longos intervalos quase em caixa.

### Bloqueios reproduzidos

- 2 anos: em 2024-12-08, `Saldo não gerenciado detectado em BTCUSDT: +0.000050000000`. O replay registrou 28 exceções diárias/de reconciliação, preservando a lógica de contenção do live.
- 3 anos: em 2024-10-05, `Saldo não gerenciado detectado em BNBUSDT: +0.009000000000`. O replay registrou 1499 exceções diárias/de reconciliação, preservando a lógica de contenção do live.
- 4 anos: em 2024-04-08, `Saldo não gerenciado detectado em BTCUSDT: +0.000070000000`. O replay registrou 1682 exceções diárias/de reconciliação, preservando a lógica de contenção do live.

O live pode eliminar a posição virtual quando uma sobra fica abaixo do mínimo negociável; se essa sobra depois valorizar e atingir o mínimo, a reconciliação a identifica como saldo não gerenciado. O halt persiste sem comando manual. A simulação não adota o saldo nem limpa o halt para fabricar continuidade. As sobras e arredondamentos dependem de filtros e comissões assumidos; as datas acima são achados do replay, não prova de ocorrências reais na conta.

Há ainda um risco contábil no código atual: a reconciliação de uma OCO desaparecida estima ativos removidos pelo preço marcado e ajusta o peak pelo fluxo líquido calculado, podendo interpretar parte de um fill/custo como fluxo externo. Além disso, o rebalanceamento entre sleeves não tem uma guarda global de halt própria. Esses comportamentos foram preservados; demandam uma correção específica no live e nova execução para avaliar a estratégia sem essas falhas.

### Walk-forward

Foram executadas janelas contíguas de 90 dias, com última janela parcial, parâmetros congelados e carteira/risco carregados entre folds. Cada decisão usa somente 399 candles fechados anteriores, reproduzindo o limite da API live. Não há seleção de TP, otimização ou ajuste após observar os folds. O retorno composto dos folds coincide com o backtest contínuo: os folds são uma decomposição temporal do mesmo replay, não um segundo experimento independente.

Os históricos anteriores à promoção da configuração em agosto/2026 são retrospectivos. Não podem ser apresentados como OOS intocado; apenas datas posteriores à seleção podem contribuir para um forward não usado na seleção, sujeito às demais limitações. Os arquivos `*_walk_forward.csv` contêm os retornos e drawdowns por janela.

## Limites de fidelidade

A configuração, o cálculo de decisões, o sizing e a reconciliação são compartilhados com o live. A execução não é idêntica à bolsa real: OPEN diário aproxima o preço das 00:05; LIMIT aceita é suposta preenchida; taxa é cobrada em USDT; filtros são os atuais; fills parciais e filas não são reproduzidos. OCO dupla no mesmo dia resolve SL primeiro. O monitoramento de 30 segundos e picos intradiários não cabem em OHLC diário. Custos/retornos não incluem tributos pessoais. Todas as hipóteses e hashes estão em `manifest.json`.

## Limpeza de branches

Foram excluídas **17 branches já mescladas**, depois de verificar novamente PR, branch, SHA e proteção. Cada SHA original foi preservado em `refs/tags/archive/2026-10-28/<nome-da-branch>`. Não foi inferida obsolescência apenas pela data ou pela contagem de commits, pois vários merges foram squash.

| Branch removida | PR mesclada | Destino |
|---|---:|---|
| `chore/remove-obsolete-files-20260904` | #22 | `main` |
| `feature/aggressive-c-spot-margin-pool` | #19 | `main` |
| `feature/client-dashboard-mvp` | #21 | `main` |
| `feature/live-aggressive-c` | #17 | `main` |
| `fix/backtest-live-safety-2026-08` | #1 | `main` |
| `fix/hard-stop-concentration-gate` | #4 | `main` |
| `fix/live-c-entrypoint` | #18 | `main` |
| `fix/manual-balance-false-drawdown` | #24 | `main` |
| `fix/manual-balance-reconciliation` | #23 | `main` |
| `fix/oco-quantity-format` | #25 | `main` |
| `fix/spot-only-manual-decision` | #20 | `main` |
| `paper/cross-asset-hybrid-v1` | #11 | `research/adaptive-portfolio-v1` |
| `research/cross-asset-allocation-v1` | #10 | `research/adaptive-portfolio-v1` |
| `research/cross-asset-opportunity` | #6 | `main` |
| `research/entry-state-forensics` | #7 | `main` |
| `research/forensic-bull-sideways` | #5 | `main` |
| `research/stages-2-6-robustness` | #3 | `main` |

Mantidas: `main` e sete branches de pesquisa com código exclusivo/PRs não incorporadas. A branch desta correção registra o trabalho atual. Evidência da execução: https://github.com/vinirex7/Krypton/actions/runs/40786606225 . Os nomes e SHAs estão em `branch_audit.json` e as tags confirmadas em `branch_cleanup_verified.json`. O workflow temporário de limpeza foi removido após a verificação.

## Verificação e reprodução

54 testes determinísticos aprovados localmente e checks GitHub aprovados na implementação inicial. A suíte adicional testa o uso real do motor live sem cliente de rede, causalidade, histórico fechado, OCO/gap, gate de preço, soma cronológica dos folds, cobertura de dados e halt por erro. `tests.txt` e `market_run.txt` preservam evidências.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python live_validation.py --as-of 2026-10-28 \
  --cache-dir reports/2026-10-28/data --output results/reproduction-20261028
```

O cache incluído tem os candles e filtros utilizados. A validação de datas/OHLC é estrita e falha se faltar cobertura. `summary.json`/`summary.csv` resumem os resultados; `*_equity.csv`, `*_orders.csv`, `*_rejections.csv`, `*_runtime_errors.csv` e `*_state.json` permitem examinar os detalhes.
