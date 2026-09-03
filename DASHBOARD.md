# Krypton Dashboard — MVP

Este MVP adiciona acompanhamento privado do contratante sem acoplar o painel ao motor de execução.

## O que existe

- Login por e-mail e senha.
- Sessões autenticadas por bearer token.
- Isolamento por `client_id` em snapshots, decisões, ordens e trades.
- Patrimônio, retorno acumulado, drawdown, modo TESTNET/PRODUÇÃO e status de halt.
- Curva de patrimônio.
- Feed de decisões táticas e alpha.
- Ordens executadas pelo motor.
- Trades táticos encerrados pelo próprio motor.
- Atualização do navegador a cada 15 segundos.
- Snapshot do bot aproximadamente a cada 60 segundos.

O banco `krypton_dashboard.db` é separado do banco de estado `krypton_c_state.db`. Falhas de telemetria são registradas em log e não devem bloquear a execução do bot.

## 1. Instalar

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

No `.env`, associe a instância do bot a um contratante:

```env
KRYPTON_CLIENT_ID=cliente_001
KRYPTON_DASHBOARD_DB=krypton_dashboard.db
```

## 2. Criar o acesso do contratante

Use o mesmo `KRYPTON_CLIENT_ID` configurado no bot:

```bash
python dashboard_admin.py --id cliente_001 --name "Cliente Krypton" --email cliente@exemplo.com
```

A senha é solicitada sem aparecer no terminal e deve ter pelo menos 10 caracteres.

## 3. Iniciar a API/painel

```bash
uvicorn dashboard_api:app --host 127.0.0.1 --port 8000
```

Em produção, publique o serviço atrás de Nginx/Caddy com HTTPS. Não exponha diretamente a porta 8000 à Internet sem TLS e controles de rede.

## 4. Iniciar o Krypton

```bash
python tradebot.py
```

O live grava a telemetria usando o `KRYPTON_CLIENT_ID` da instância.

## Segurança

- Nunca armazene API key/secret da Binance no banco do dashboard.
- Nunca habilite saque na API Binance.
- Use HTTPS antes de fornecer o painel a terceiros.
- O banco do dashboard contém hashes de senha e dados financeiros; mantenha permissões de arquivo restritas e backups protegidos.
- O front-end nunca recebe dados de outro `client_id`; a filtragem ocorre no backend autenticado.

## Limitações intencionais do MVP

- Atualização via polling de 15 s, ainda sem WebSocket.
- Saídas táticas executadas diretamente pelo motor entram em `trades`; fills de OCO detectados apenas por reconciliação ainda não possuem preço de fill histórico no painel.
- O motivo público da decisão é propositalmente resumido para não expor thresholds e regras proprietárias da estratégia.
- O MVP usa SQLite. Para vários contratantes/instâncias concorrentes, a evolução recomendada é PostgreSQL + API centralizada.
