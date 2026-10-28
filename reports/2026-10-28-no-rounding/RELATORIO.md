# Krypton — mesma estratégia, ordens sem arredondamento

Experimento solicitado em 28/10/2026. Base: `main`, commit `7cc2d4a527066c78c9ca55c63200a9dd0b780c71`. Não altera o live, a configuração da estratégia nem o comportamento padrão do replay.

## O que mudou

Somente a bolsa simulada deixou de quantizar quantidades por `stepSize` e preços por `tickSize`, inclusive os níveis e a quantidade da OCO. Os números continuam em float64; isso não significa aritmética exata ou precisão arbitrária. Valores do relatório são formatados para leitura.

Sinais, perfil 55/45, risco tático de 2%, regras alpha, gates, mínimo financeiro por ordem, verificações de saldo, taxas, slippage, critérios de dust, reconciliação e halts são os mesmos. O fingerprint, perfil, configuração de execução e proveniência dos dados foram comparados por igualdade com o experimento anterior. Os módulos live, cliente Binance, indicadores, configuração e gestão de risco permaneceram inalterados.

## Resultados

Capital inicial independente de 10.000 USDT por período, dados até 27/10/2026 inclusive. Patrimônio final marcado a mercado, sem liquidação forçada.

| Período | Retorno anterior, com arredondamento | Retorno sem arredondamento | Anualizado sem arredondamento | DD máximo diário | Capital final USDT | Folds WF |
|---|---:|---:|---:|---:|---:|---:|
| 10 meses | +1.0406% | +1.0418% | 1.2530% | -7.8497% | 10104.1786 | 4 |
| 2 anos | +19.9864% | +18.4892% | 8.8591% | -21.4012% | 11848.9162 | 9 |
| 3 anos | +50.3216% | +77.1812% | 21.0005% | -20.4571% | 17718.1211 | 13 |
| 4 anos | +66.4390% | +77.8812% | 15.4868% | -21.8263% | 17788.1181 | 17 |

Os retornos anteriores de 2, 3 e 4 anos incluíam longos bloqueios operacionais por sobras de arredondamento. Quando esses bloqueios desaparecem, há novas negociações, custos e exposição ao mercado. A diferença entre resultados inclui essa mudança de trajetória, não apenas centavos economizados no preço de cada ordem.

## Estado de risco e execução

| Período | Exceções do runtime | Primeiro halt global | Halt tático ao final | Quantidade de ordens preenchidas | Taxas USDT |
|---|---:|---|---|---:|---:|
| 10 meses | 0 | Nenhum | False | 39 | 54.4697 |
| 2 anos | 0 | Nenhum | False | 245 | 308.7491 |
| 3 anos | 0 | Nenhum | False | 380 | 668.0910 |
| 4 anos | 0 | Nenhum | False | 443 | 719.8266 |

Os arquivos de erros preservam qualquer falha ocorrida; as proteções de risco não foram removidas para produzir estes resultados.

## Walk-forward e limitações

Janelas contíguas de 90 dias, com janela final parcial e parâmetros congelados. O estado, incluindo risco e halts, passa entre janelas. O retorno composto dos folds coincide com o replay contínuo, por construção. Períodos anteriores à seleção da estratégia em agosto/2026 são retrospectivos, não OOS intocado.

Este é um contrafactual: a Binance real exige quantidades e preços compatíveis com seus filtros. Os fills continuam aproximados com candles diários e não reproduzem filas ou o monitoramento a cada 30 segundos. O experimento isola a quantização das ordens, não corrige outros comportamentos operacionais do live.

## Reprodução e evidências

```bash
python -m unittest discover -s tests -v
python live_validation.py --as-of 2026-10-28 --no-rounding \
  --cache-dir reports/2026-10-28/data \
  --output results/2026-10-28-no-rounding
```

56 testes determinísticos aprovados. `manifest.json` registra a configuração, o modo sem arredondamento e hashes; os dados e filtros são exatamente os preservados em `../2026-10-28/data/`. Curvas, ordens, rejeições, erros, estados e folds estão nesta pasta.
