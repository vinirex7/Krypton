# Krypton · K1316

Robô short de BTCUSDT perpétuo na Binance USDⓈ-M Futures. Usa candles de 4h do contrato e taxas de funding liquidadas. O comando padrão consulta o sinal usando dados públicos. A execução de ordens exige escolher `demo` ou `live` e configurar as credenciais correspondentes.

## Instalação e consulta

Em Linux ou macOS, com Python 3.11 ou 3.12, execute na pasta do repositório:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python krypton.py signal
```

A consulta imprime o horário UTC de abertura do candle de entrada, o módulo e o ATR do sinal. `None` indica ausência de sinal. A consulta durante uma vela mostra a decisão referente à abertura daquela vela.

## Conta demo da Binance

Use uma conta Futures dedicada, com saldo em USDT, BTCUSDT em **One-way Mode**, **Single-Asset Mode** e alavancagem configurada em **2×**. O robô dimensiona a posição pelo saldo USDT da carteira. Depósitos, retiradas e operações manuais durante a execução alteram o cálculo de drawdown. O programa exige Linux ou macOS, relógio sincronizado, conexão contínua e um único processo por conta.

Crie as chaves na conta demo de USDⓈ-M Futures. Disponibilize-as ao processo nas variáveis de ambiente `KRYPTON_DEMO_KEY` e `KRYPTON_DEMO_SECRET`. Use um diretório persistente exclusivo da conta:

```bash
export KRYPTON_STATE_DIR="$HOME/.local/share/krypton"
python krypton.py demo
```

A demo usa `https://demo-fapi.binance.com`. Confira no histórico da conta o preenchimento da venda, o stop BUY, o alvo BUY para o módulo de funding, o encerramento por prazo e a retomada do processo com o mesmo diretório de estado. O modo demo opera o mercado de demonstração, cujos preços e funding podem diferir dos da produção.

## Conta real

Após validar a execução na demo, configure `KRYPTON_LIVE_KEY` e `KRYPTON_LIVE_SECRET` no ambiente privado do processo. A chave precisa de leitura e negociação de Futures. Guarde-a fora do repositório. O comando abaixo roda continuamente e pode enviar ordens com dinheiro real na próxima abertura elegível.

```bash
python krypton.py live
```

O bot consulta a Binance a cada ciclo. Só inicia uma posição nos primeiros 30 segundos da vela, usando o preço disponível. Mantém o pico de patrimônio, a última vela processada e a intenção de ordem em SQLite. Demo e produção têm arquivos distintos. Preserve esses arquivos nos reinícios. Rode uma única instância em uma única máquina por conta. Ao interromper o processo, os gatilhos já aceitos ficam na corretora. Saídas por prazo e atualizações do trailing dependem do processo em funcionamento.

O processo reconcilia uma venda com resposta incerta pelo identificador original. Uma posição incompatível com o estado requer conferência da conta. Se a Binance rejeitar o primeiro stop ou a instalação ficar incerta, o programa tenta recomprar a posição a mercado com `reduceOnly`. Falhas de rede podem atrasar tanto a proteção quanto essa recompra. Se houver divergência, confira posições, ordens normais, ordens condicionais e o histórico de execuções da conta antes de retomar. Mantenha o banco de estado para a reconciliação.

## Regra congelada

O sinal é calculado no fechamento. A K1316 espera uma vela completa adicional e vende na abertura seguinte, com uma posição por vez. O ATR é o da vela de sinal. A precedência dos módulos é a ordem da tabela.

| Módulo | Entrada | Saída |
|---|---|---|
| Funding | Tendência de baixa, EMA200 descendente em 30 velas, preço abaixo da EMA21, abaixo do fechamento de 6 velas atrás, candle negativo e z-score de funding acima de 0,5 | Stop 2,5 ATR, alvo 3 ATR ou 30 velas |
| Compressão | Preço e EMA50 abaixo da EMA200, largura das bandas de Bollinger no quintil inferior de 270 velas, fechamento abaixo da banda inferior | Stop inicial 2,5 ATR, recalculado a cada abertura pelo menor preço anterior da posição mais 3 ATR, limitado pelo stop inicial |
| Candle de baixa | Corpo baixista maior que 1 ATR, fechamento nos 15% inferiores da amplitude e acima da EMA200 | Stop 2,5 ATR ou 12 velas |

O funding usa a média de três liquidações e o z-score de 270 observações, com horário de publicação até o fechamento do sinal. A tendência de baixa exige preço e EMA50 abaixo da EMA200. O tamanho busca risco de 3% do patrimônio até o stop, limitado a 2× o saldo em exposição nominal. Em drawdown de 15%, 25% e 35%, o multiplicador passa a ½, ¼ e zero. Em execução, o tamanho também respeita margem disponível, reserva para taxas e filtros de quantidade e preço da Binance.

## Reprodução histórica

Para usar o pacote de dados do projeto, instale `pyarrow` no ambiente e indique o diretório que contém o parquet de BTC perpétuo e o CSV de funding. O script compara operações, retorno e drawdown com os valores arquivados em três cenários.

```bash
python -m pip install pyarrow
python verify_history.py /caminho/Krypton_Dados_Reproducao
```

| Intervalo até 22/11/2026 | Custos | Operações | Retorno líquido | DD intrabar estimado |
|---|---:|---:|---:|---:|
| Desde 01/12/2019 | Base | 238 | +149,00% | 17,30% |
| Desde 01/01/2024 | Base | 115 | +52,01% | 17,25% |
| Desde 01/01/2024 | Dobrados | 115 | +23,36% | 19,72% |

O replay usa comissão de 0,05% e slippage de 0,02% por lado no cenário-base, funding histórico, preços OHLC e a convenção de stop primeiro quando duas barreiras são tocadas na mesma vela. A conta real usa preços de preenchimento, taxas efetivas, latência e filtros da Binance. O executor inicializa os indicadores com até 1.500 candles recentes. O replay usa a série integral. A taxa de funding e as oscilações entre leituras podem gerar pequenas diferenças na avaliação do patrimônio. O resultado histórico ainda requer validação prospectiva.

### Reinício automático no macOS (demo)

Com o processo demo anterior parado, use o mesmo diretório de estado SQLite.
Se ele estiver fora de `var` na pasta do projeto, exporte `KRYPTON_STATE_DIR`
com o caminho absoluto antes de instalar. No arquivo privado
`~/.krypton-demo.env` coloque as atribuições shell de `KRYPTON_DEMO_KEY` e
`KRYPTON_DEMO_SECRET` e aplique `chmod 600 ~/.krypton-demo.env`.
No ambiente Python do Krypton execute `python install_macos_demo.py`.
O serviço inicia imediatamente, volta após encerramento e carrega no login.
O Mac precisa estar acordado. Consulte `demo.log` no diretório de estado.
Para parar e remover, execute
`launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.krypton.demo.plist`
e remova esse plist. Divergências de posição ou ordens exigem conferência.
