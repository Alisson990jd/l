# FutData updater

Mantém o dataset [Alisson990/futdata](https://huggingface.co/datasets/Alisson990/futdata) atualizado.
O workflow `.github/workflows/futdata-daily.yml` roda todo dia às 06:17 UTC (e manualmente em *Actions → Run workflow*).

| Módulo | Fonte | O que atualiza |
|---|---|---|
| `sportmonks.py` | Sportmonks (plano grátis) | Premier League, Bundesliga, Superliga DK, Premiership SCO: partidas, eventos, estatísticas por jogador e por time, lesões |
| `api_football.py` | API-Football | jogos do dia no mundo todo e detalhes (eventos, escalações, estatísticas por jogador), lesões |
| `open_sources.py` | football-data.co.uk, martj42, transfermarkt-datasets | temporada atual das ligas europeias, jogos de seleções, Transfermarkt (às segundas) |
| `readme.py` | — | regenera o README do dataset |

## Secrets necessários (Settings → Secrets and variables → Actions)

- `HF_TOKEN`: token do Hugging Face com permissão de escrita
- `SPORTMONKS_TOKEN`: token da API Sportmonks
- `APIFOOTBALL_KEY` (opcional): chave da API-Football

Sem um secret, a etapa correspondente é pulada.

## Rodar localmente

```bash
pip install -r futdata/requirements.txt
export HF_TOKEN=... SPORTMONKS_TOKEN=... FUTDATA_OUT=out
python -m futdata.sportmonks 2026-08-01 2026-11-17   # período específico
python -m futdata.readme
python -c "from futdata.common import upload; upload('manual update')"
```
