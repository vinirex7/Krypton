# 🔷 Krypton TradeBot

> **Estratégia Vencedora: Supertrend + RSI + MACD Filter**  
> Backtest: Jan/2022 – Mai/2026 · 1.612 dias · Capital simulado: $10.000 USDT

---

## 📊 Performance (Backtest)

| Métrica | Bot (SOLUSDT) | Buy & Hold SOL |
|---|---|---|
| **Retorno Total** | **+37,7%** | -53,9% |
| **Sharpe Ratio** | **0,932** | — |
| **Max Drawdown** | **-5,0%** | -94,4% |
| **Win Rate** | **54,0%** | — |
| **Profit Factor** | **1,748** | — |
| **Nº de Trades** | **74** | — |
| **Alpha vs B&H** | **+91,6 p.p.** | — |

**Portfolio Multi-Ativo** (BTC 40% | ETH 20% | SOL 25% | BNB 15%): +18,5% vs B&H +4,0%

---

## 🧠 Estratégia: Confirmação Tripla

A estratégia exige que **três filtros concordem simultaneamente** antes de abrir uma posição:

| # | Indicador | Condição LONG | Condição SHORT |
|---|---|---|---|
| 1 | **Supertrend** (period=7, mult=3.0) | Direção = +1 (bullish) | Direção = -1 (bearish) |
| 2 | **RSI** (period=14) | 40 ≤ RSI ≤ 70 | 30 ≤ RSI ≤ 60 |
| 3 | **MACD** (12,26,9) | MACD line > Signal line | MACD line < Signal line |

> O filtro assimétrico do RSI (40–70 para long, 30–60 para short) é resultado da otimização via grid search.

---

## 🛡️ Gestão de Risco

| Regra | Valor | Descrição |
|---|---|---|
| Risco por Trade | **1%** do capital | Máxima perda aceita por trade |
| Stop Loss | **2,0× ATR(14)** | Stop automático baseado em volatilidade |
| Take Profit | **3,0× ATR(14)** | Ratio R:R = 1,5:1 |
| Circuit Breaker | **-4%** no dia | Fecha tudo + pausa 24h |
| Max Drawdown Stop | **-20%** total | Halt completo → requer intervenção manual |
| Posições simultâneas | **máx 4** | 1 por par |
| Tipo de ordem | **Limit only** | Nunca market orders |

**Position Sizing (ATR-based):**
```
Quantity = (Capital × 1%) / (ATR(14) × 2.0)
```

---

## 📁 Estrutura do Projeto

```
Krypton/
├── config.py          ← API keys, parâmetros da estratégia, gestão de risco
├── indicators.py      ← RSI, MACD, ATR, Supertrend, geração de sinais
├── risk_manager.py    ← Position sizing, circuit breaker, max drawdown
├── binance_client.py  ← Interface com API da Binance (dados + ordens)
├── tradebot.py        ← Loop principal de trading (00:05 UTC diário)
├── backtest.py        ← Script standalone de backtesting
├── requirements.txt   ← Dependências Python
├── .env.example       ← Template de variáveis de ambiente
├── .gitignore         ← Inclui .env, logs, __pycache__
└── README.md          ← Este arquivo
```

---

## 🚀 Deploy (Passo a Passo)

### 1. Pré-requisitos

- Python 3.11+
- Conta Binance com KYC verificado
- VPS (recomendado: AWS `sa-east-1` São Paulo para menor latência)

### 2. Configurar API Binance

1. Acesse **Binance → Perfil → Gerenciamento de API**
2. Criar nova API key
3. Habilitar **apenas**: `Enable Reading` + `Enable Spot & Margin Trading`
4. ❌ **NUNCA** habilitar saques ou futuros
5. Restringir ao IP fixo do seu VPS

### 3. Instalar e Configurar

```bash
# Clonar o repositório
git clone https://github.com/vinirex7/Krypton.git
cd Krypton

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\\venv\\Scripts\\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar credenciais
cp .env.example .env
nano .env  # Adicionar BINANCE_API_KEY e BINANCE_API_SECRET
```

### 4. Testar com Backtest

```bash
# Replicar o backtest original (SOLUSDT 2022–2026)
python backtest.py --symbol SOLUSDT --start 2022-01-01 --end 2026-06-01

# Testar outros pares
python backtest.py --symbol BTCUSDT --start 2022-01-01
python backtest.py --symbol BNBUSDT --start 2022-01-01
```

### 5. Executar em Testnet (Obrigatório: mínimo 30 dias)

```bash
# config.py já vem com USE_TESTNET = True
python tradebot.py
```

### 6. Produção (apenas após testnet estável)

```bash
# Em config.py: USE_TESTNET = False
# Capital inicial recomendado: $500 USDT

# Executar em background persistente (screen)
screen -S krypton
python tradebot.py
# Ctrl+A, D → desconecta sem parar o bot

# Reconectar
screen -r krypton

# Ver logs em tempo real
tail -f tradebot.log
```

---

## 📈 Pares e Alocação

| Par | Alocação | Motivo |
|---|---|---|
| **SOLUSDT** | 25% | Melhor performance — alpha +91,6 p.p. vs B&H |
| **BTCUSDT** | 40% | Maior liquidez, menor risco |
| **ETHUSDT** | 20% | Diversificação |
| **BNBUSDT** | 15% | Desconto em taxas Binance |

---

## 🔍 Monitoramento

```bash
# Log em tempo real
tail -f tradebot.log

# Filtrar apenas alertas críticos
grep -E "CIRCUIT|MAX DRAWDOWN|ERROR" tradebot.log

# Contar trades por dia
grep "Nova posição" tradebot.log | cut -d'|' -f1 | sort | uniq -c
```

---

## ⚠️ Avisos Importantes

> **RISCO DE PERDA TOTAL:** Trading algorítmico de criptoativos envolve risco substancial. É possível perder 100% do capital.

1. **Performance passada não garante resultados futuros** — backtest é simulação.
2. **Capital de risco apenas** — nunca investir mais do que pode perder.
3. **Testnet é obrigatório** — mínimo 30 dias antes de produção.
4. **Sem alavancagem** — nunca usar dinheiro emprestado para capitalizar o bot.
5. **Supervisão manual** — revisar logs semanalmente, especialmente no início.
6. **Conformidade legal** — verificar regulação de trading algorítmico na sua jurisdição.

---

## 🔧 Manutenção

- **Semanal:** Revisar `tradebot.log` para comportamentos inesperados
- **Mensal:** Backtest com dados novos para verificar performance
- **Trimestral:** Atualizar `requirements.txt` (segurança)
- **Monitorar:** Drawdown atual — se > 15%, revisar parâmetros

---

## 📄 Licença

Este projeto é disponibilizado para fins **educacionais e de pesquisa**. O autor não se responsabiliza por perdas financeiras decorrentes do uso deste software.

---

*Relatório técnico completo: `TradeBot_Cripto_Binance.pdf` (backtest Jan/2022–Mai/2026)*
