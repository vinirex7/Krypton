# Krypton · KSO1316, KSL1 e KSL2

Robô de BTC perpétuo com execução na Binance USDⓈ-M ou na Kraken Derivatives. Usa candles de 4h do contrato e taxas de funding liquidadas. O comando padrão consulta o sinal usando dados públicos. A execução de ordens exige escolher `demo` ou `live` e configurar as credenciais correspondentes.

## Teste inicial com R$ 100 na Kraken

Com o Python e as dependências instalados, execute no diretório do projeto:

```bash
python krypton.py signal --exchange kraken --strategy KSL2
python krypton.py demo --exchange kraken --strategy KSL2
```

Para a segunda linha, exporte localmente `KRYPTON_KRAKEN_DEMO_KEY` e `KRYPTON_KRAKEN_DEMO_SECRET` de uma conta **demo**. O serviço começa a monitorar os candles e só entra se surgir um sinal e o tamanho calculado alcançar o lote mínimo. O primeiro início exige uma carteira demo dedicada em USD sem posições ou ordens e um estado novo. Preserve o estado nos reinícios. O orçamento inicial é R$ 100 dividido pela última cotação diária de venda do dólar do Banco Central (SGS 1), arredondado para baixo em centavos de USD. O robô registra esse valor e separa o restante do saldo virtual como reserva. Lucros e perdas posteriores alteram o capital do teste. Mantenha a carteira sem depósitos, saques ou negociações externas durante a demo. Nenhuma conversão de moeda ou transferência é executada.

## Operação conjunta em candles de 4 horas

Escolha `--strategy KSL2` para combinar a KSO1316 com a compra por rompimento. A cada abertura, com a conta sem posição, o bot verifica primeiro os três módulos SHORT da KSO1316 na ordem original e com sua espera de uma vela. Quando nenhum deles sinaliza, compra se o último fechamento superar a maior máxima das **55 velas anteriores**, o preço e a EMA50 estiverem acima da EMA200, e a EMA200 estiver acima de seu valor de 30 velas atrás. Em qualquer outro caso, aguarda. Uma posição aberta segue suas próprias saídas, mesmo se a tendência mudar.

A KSL1 acrescenta uma confirmação ao SHORT no fechamento da vela de espera. Exige preço e EMA50 abaixo da EMA200, ou preço abaixo de uma EMA21 em queda. A compra é idêntica nas duas variantes. O risco nominal da compra é 1% do patrimônio até o stop inicial de 2,5 ATR. O alvo fica a 6 ATR e o prazo máximo é 60 velas (10 dias). SHORT usa o risco original de 3%. Os dois lados compartilham o mesmo capital, teto de exposição e reduções por drawdown.

Com o ambiente do projeto ativo:

```bash
python krypton.py signal --exchange kraken --strategy KSL2
python krypton.py demo --exchange kraken --strategy KSL2
```

Para iniciar essa mesma variante automaticamente no Mac, o instalador aceita `python install_macos_demo.py --exchange kraken --strategy KSL2`. O modo demo requer as credenciais descritas abaixo. As variantes KSL estão em avaliação em sinal e demo. O padrão dos comandos antigos continua sendo KSO1316. Uma operação aberta deve ser retomada com a mesma estratégia. A identificação antiga K1316 continua válida para recuperar ordens já enviadas.

`backtests_ksl.csv` contém os resultados por estratégia e período. As versões KSL foram comparadas no mesmo histórico já utilizado durante o desenvolvimento. A KSL2 foi definida após a avaliação da KSL1.

Para reproduzir, instale também `pyarrow` e execute:

```bash
python verify_history.py ../research_inputs/Krypton_Dados_Reproducao --compare
```

Cada janela começa com capital 1 e caixa livre, com indicadores aquecidos no histórico anterior. O replay usa candles e funding do BTCUSDT perpétuo da Binance, 0,05% de taxa e 0,02% de slippage por lado no cenário base. O cenário 2× dobra esses dois custos. O stop tem prioridade se stop e alvo caem na mesma vela. O drawdown segue a convenção intrabar do benchmark original. A sequência real de preços dentro de cada vela é desconhecida. O simulador encerra a posição no fim da janela e abstrai tamanho de lote, latência, liquidação e preenchimentos parciais.

### Revisão dos resultados da KSL2

O arquivo `backtests_ksl.csv` guarda o benchmark original. A auditoria é reproduzível com
`python audit_ksl2.py ../research_inputs/Krypton_Dados_Reproducao auditoria_ksl2.csv`.
Ela usa os mesmos sinais e um segundo motor de posição, custos, funding e contabilidade.
Na saída, `executor_aproximado=False` reproduz o modelo publicado. Com `True`, o teste exige
o aquecimento da API (mínimo de 1.000 candles e 273 taxas), usa até 1.500 candles e 120 dias
de funding para os sinais, limita a exposição a 1,99× o saldo livre presumido e dimensiona à abertura
antes de aplicar o slippage. A saída por prazo ocorre na abertura seguinte, inclui eventual funding
nesse limite e consome a oportunidade de entrada da mesma abertura. Os períodos começam sem posição.

Ambos os cenários usam BTCUSDT da Binance e o mesmo custo por lado. Quando falta o mark price de
funding, usam a abertura do candle. O drawdown intrabar é o da convenção do benchmark, que depende
da ordem presumida de preços e pode ignorar excursões anteriores à execução de uma barreira.
A auditoria fornece também o drawdown medido somente nos fechamentos. Lotes mínimos, precisão de preço,
livro de ordens, margem real, liquidação, falhas de rede e liquidações de funding da Kraken não foram simulados.
A fração `contribuicao_long` é o P&L líquido acumulado de compras dividido pelo capital inicial da janela.
O treino e as revisões passadas já consultaram esses períodos.

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

## Kraken Derivatives no Mac

O contrato de execução é o perpétuo linear BTC/USD (`PF_XBTUSD`), na aba **Perps → BTC Perp**. Os candles de referência de 4 horas, os mínimos usados no trailing, o ATR e o funding dos sinais continuam vindo do BTCUSDT perpétuo da Binance por API pública. A Kraken fornece o preço de execução, os preenchimentos, o saldo em USD da carteira de derivativos e o funding efetivamente cobrado ou recebido. Diferenças de preço e de custos entre as corretoras afetam as saídas e o resultado.

Na pasta do projeto, com o ambiente virtual ativo, consulte o sinal e execute os testes sem credenciais:

```bash
python krypton.py signal --exchange kraken
python -m unittest discover -s tests -v
```

Para testar ordens, use credenciais de um ambiente de demonstração Kraken Derivatives habilitado. O modo `demo` usa exclusivamente `demo-futures.kraken.com` para a conta e ordens. A disponibilidade de acesso depende da Kraken. Configure uma carteira de derivativos separada e dedicada ao bot e o limite fixo de 2× para BTC Perp. A chave precisa de leitura e negociação de derivativos. Mantenha a permissão de saques desabilitada. O saldo da carteira e a margem livre são expressos em USD. Prefira garantia em USD para a comparação com o histórico, pois outra garantia também oscila de preço.

Guarde as atribuições `KRYPTON_KRAKEN_DEMO_KEY='...'` e `KRYPTON_KRAKEN_DEMO_SECRET='...'` no arquivo privado `~/.krypton-kraken-demo.env` com permissão 600. Para iniciar em primeiro plano:

```bash
set -a
source ~/.krypton-kraken-demo.env
set +a
python krypton.py demo --exchange kraken
```

O estado e o bloqueio ficam em `var/kraken/` ou em `$KRYPTON_STATE_DIR/kraken/`. Use o mesmo diretório após reiniciar. Para iniciar automaticamente a demo ao fazer login no macOS, encerre a execução manual e rode `python install_macos_demo.py --exchange kraken`. O serviço se chama `com.krypton.kraken.demo` e grava em `kraken-demo.log` no diretório de estado base.

Valide na demo o preenchimento, a proteção da quantidade negociada, o encerramento por prazo e a retomada após desconexão. Em caso de confirmação ambígua o robô consulta o identificador original e o histórico da conta. O stop e o alvo de recompra usam o último preço negociado e `reduceOnly`. Ordens a mercado da Kraken podem ter preenchimento parcial. Ao interromper o bot, os gatilhos aceitos ficam na corretora e o trailing e a saída por prazo dependem da retomada.

O comando de produção é `python krypton.py live --exchange kraken`. Ele usa `KRYPTON_KRAKEN_LIVE_KEY` e `KRYPTON_KRAKEN_LIVE_SECRET` e pode enviar ordens reais. A tabela de reprodução histórica abaixo usa dados e custos da Binance. A validação de execução autenticada e dos resultados prospectivos na Kraken é uma etapa separada.

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

O sinal é calculado no fechamento. A KSO1316 espera uma vela completa adicional e vende na abertura seguinte, com uma posição por vez. O ATR é o da vela de sinal. A precedência dos módulos é a ordem da tabela.

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
