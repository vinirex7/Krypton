# Krypton — teste dos últimos seis anos e verificação dos números publicados

## Conclusão

O log histórico da pesquisa candidata C contém exatamente os valores arredondados anunciados na imagem: +397,77%, CAGR 30,46% e drawdown -22,15%. A origem documental desses números está confirmada. Isso não é uma reprodução independente daquele teste e não constitui um registro de operação real.

O teste novo usa o motor live atual, sem quantização das ordens, no período de **28/10/2020 a 27/10/2026**, com 10.000 USDT iniciais. O resultado foi **+290.9742%**, CAGR **25.5203%** e drawdown máximo diário **-24.1667%**.

## Comparação

| Métrica | Pesquisa C arquivada | Novo replay do motor atual |
|---|---:|---:|
| Datas solicitadas | 01/08/2020–25/08/2026 | 28/10/2020–27/10/2026 |
| Retorno acumulado | +397,7737% | +290.9742% |
| Retorno anualizado composto (CAGR) | 30,4550% | 25.5203% |
| Drawdown máximo | -22,1542% | -24.1667% |
| Sharpe | 1,255064 | 1.191590 |
| Patrimônio final, base 10.000 USDT | 49.777,366718 | 39097.419510 |

As datas são diferentes e os simuladores também. Portanto, a diferença não pode ser atribuída exclusivamente a um deles. Não há base para afirmar que os números da imagem representam o desempenho validado do motor atual. O rótulo da imagem “retorno médio ao ano” corresponde, neste log, a CAGR, não a média aritmética dos retornos anuais.

## Fonte exata dos números antigos

Workflow: https://github.com/vinirex7/Krypton/actions/runs/33115971738

Job: https://github.com/vinirex7/Krypton/actions/runs/33115971738/job/98670591532

Comando histórico: `python c_only_pre_live_validation.py --start 2020-08-01 --end 2026-08-25 --mc-runs 5000`, executado em 27/08/2026.

Linha `C` da tabela `FULL`: retorno `3.977737`, CAGR `0.304550`, Sharpe `1.255064`, drawdown `-0.221542`, capital final `49777.366718`.

Os trechos exatos estão em `published_numbers_evidence.json`. O código antigo simula separadamente tático e alpha e depois combina suas curvas (`combine_rebalanced_sleeves`). O replay novo utiliza os métodos de decisão, execução e reconciliação do live com uma bolsa simulada e caixa compartilhado. Ele também replica a janela móvel de candles do live. Esses motores não são intercambiáveis apenas porque seus parâmetros declarados são iguais.

## Configuração e execução do teste novo

- Mesmo perfil Aggressive C: 55% tático/45% alpha, risco de 2% da equity tática, stop 2 ATR, TP 3 ATR, alpha com alvo de volatilidade de 30%, cadências 45/90 dias.
- Sem arredondamento de quantidades por lote ou preços por tick, incluindo OCO. Taxas, slippage, mínimo financeiro, sinais, gates e regras de risco preservados.
- Os arquivos live, cliente Binance, indicadores, configuração e risco não foram alterados. O fingerprint e a configuração de execução coincidem com o experimento anterior.
- Capital inicial de 10.000 USDT em caixa; final marcado a mercado, sem venda final forçada.
- Ordens preenchidas: 613; taxas simuladas: 1844.200730 USDT.
- Exceções do runtime: 0; primeiro halt global: nenhum; halt tático no estado final: False.

## Cobertura e preparação dos indicadores

SOL/USDT tem histórico oficial desde 11/08/2020. Em 28/10/2020 havia apenas 78 candles diários fechados disponíveis para SOL, contra até 399 dos demais ativos. Foi utilizada a opção explícita `--allow-short-warmup`: o histórico anterior começa na listagem verificada na API, sem fabricar preços e sem permitir lacunas durante os seis anos efetivamente testados. Filtros que exigem mais histórico aguardam esse histórico pelas próprias funções do live.

Dados e filtros usados estão em `data/`; `manifest.json` registra datas, hashes, perfil e hipóteses. O novo resultado depende dessa inicialização em caixa e da disponibilidade real de histórico naquela data.

## Walk-forward

Foram avaliadas 25 janelas contíguas de 90 dias, com última janela parcial, parâmetros fixos e estado contínuo. O retorno composto foi +290.9742%, igual ao replay contínuo por construção. Não houve treinamento, seleção de TP ou otimização nos folds. Trata-se de decomposição temporal da política fixa, não de um segundo teste independente. O histórico anterior à seleção da estratégia em agosto/2026 é retrospectivo, não OOS intocado.

## Limites e reprodução

A Binance real exige lotes e ticks válidos; remover a quantização é um contrafactual. Candles diários aproximam o preço das 00:05 e os fills, não reproduzem filas, fills parciais nem o monitoramento a cada 30 segundos. O drawdown é medido sobre equity diária e pode não capturar quedas intradiárias maiores. Custos pessoais/tributos não estão incluídos.

58 testes determinísticos aprovados, incluindo autorização explícita para histórico curto e rejeição de lacunas dentro da janela de execução.

```bash
python -m unittest discover -s tests -v
python live_validation.py --as-of 2026-10-28 --start 2020-10-28 --end 2026-10-27 \
  --no-rounding --allow-short-warmup \
  --cache-dir reports/2026-10-28-six-year/data --output results/six-year-reproduction
```

`summary.json`, curva, ordens, rejeições, erros, estado final e folds estão nesta pasta. O live e o site não foram modificados por este experimento.
