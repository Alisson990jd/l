"""Regenerate the dataset card (README.md) from the files currently on the Hub plus the local OUT folder."""
import datetime as dt, os
from huggingface_hub import HfApi
from .common import REPO, OUT

DESC={
"api_football":("API-Football via [eatpizzanot/soccer-dataset](https://huggingface.co/datasets/eatpizzanot/soccer-dataset), CC-BY-4.0","2008–2026, 271 ligas","674 mil partidas, 182 mil jogadores, ~10 milhões de participações jogador-partida com **faltas cometidas/sofridas, amarelos, vermelhos**, minutos, chutes, passes, dribles; estatísticas por time; escalações; odds"),
"transfermarkt":("[transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)","2006–2026","partidas, eventos (gols, cartões, substituições), escalações, aparições por jogador, valores de mercado, transferências"),
"transfermarkt_extra":("[salimt/football-datasets](https://github.com/salimt/football-datasets)","~92 mil jogadores","**lesões** (motivo, datas, dias/jogos perdidos), perfis, seleções, histórico de valor"),
"statsbomb":("[StatsBomb Open Data](https://github.com/statsbomb/open-data)","Copas 1958–2022, Euros, Messi na La Liga 2004–2021, ligas femininas…","**cada evento de cada jogo** (passes, chutes com xG, faltas, cartões, dribles) com coordenadas; escalações com cartões"),
"wyscout_2017_18":("[Wyscout / Pappalardo et al.](https://figshare.com/collections/4415000), CC-BY-4.0","2017/18 top-5 ligas, Copa 2018, Euro 2016","3,2 milhões de eventos com coordenadas e tags (faltas, cartões, duelos)"),
"impect_bundesliga_2023_24":("[Impect Open Data](https://github.com/ImpectAPI/open-data)","Bundesliga 2023/24","939 mil eventos, KPIs por jogador por partida (packing, xG)"),
"historical_results":("[schochastics/football-data](https://github.com/schochastics/football-data), ODC-BY","1888–2023, 207 ligas","1,3 milhão de partidas; gols com minuto; incidentes (cartões, substituições, gols) de ~816 mil eventos"),
"english_football_1888":("[jfjelstul/englishfootball](https://github.com/jfjelstul/englishfootball), CC-BY-SA-4.0","1888–2024","todos os jogos da Premier League e EFL, classificações"),
"brazil_cbf":("[BrazilianFootball/Data](https://github.com/BrazilianFootball/Data) (súmulas CBF), MIT","2013–2025","Séries A–D e Copa do Brasil: jogos, gols, cartões, substituições, relacionados por partida"),
"world_cup":("[jfjelstul/worldcup](https://github.com/jfjelstul/worldcup)","1930–2022","Copas: partidas, gols, cartões, substituições, elencos, técnicos, árbitros"),
"international":("[martj42/international_results](https://github.com/martj42/international_results)","1872–2026","todos os jogos de seleções, autores dos gols, pênaltis"),
"football_data_uk":("[football-data.co.uk](https://www.football-data.co.uk)","1993–2026","estatísticas por partida (faltas, cartões, chutes, escanteios), odds"),
"football_charts":("[football-charts](https://huggingface.co/datasets/Damir81/football-charts-results-goal-timing), CC-BY-4.0","2022–2026, 93 ligas","resultados com minuto de cada gol"),
"openfootball":("[openfootball/football.json](https://github.com/openfootball/football.json), CC0","2010–2026","resultados das principais ligas europeias"),
"fantasy_premier_league":("[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)","2016–2026","Premier League jogador por rodada: minutos, gols, assistências, cartões, xG; partidas do Understat"),
}
DESC.update({
"sportmonks":("[Sportmonks](https://www.sportmonks.com) API (plano grátis), atualização diária","Premier League, Bundesliga, Superliga DK, Premiership SCO — 2025/26 e 2026/27","partidas, eventos (gols, cartões, substituições), ~75 estatísticas por jogador por jogo (faltas, cartões, nota, passes, duelos, chutes), estatísticas por time, lesões/suspensões"),
"api_football_live":("[API-Football](https://www.api-football.com), coleta diária","todas as ligas do dia (~1.100)","resultados do dia, eventos, escalações, estatísticas por jogador e time, lesões"),
})


def main():
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    local = [os.path.relpath(os.path.join(r, f), OUT) for r, _, fs in os.walk(f"{OUT}/data") for f in fs if f.endswith(".parquet")]
    files = sorted(set(files) | set(local))
    cfgs = "\n".join(f'- config_name: {p[5:-8].replace("/", "__")}\n  data_files: "{p}"' for p in files)
    groups = {}
    for p in files:
        groups.setdefault(p.split("/")[1], []).append(f"`{p.split('/')[-1][:-8]}`")
    tbl = "\n".join(f"| `{g}/` | {DESC.get(g, ('', '', ''))[0]} | {DESC.get(g, ('', '', ''))[1]} | {DESC.get(g, ('', '', ''))[2]} |" for g in groups)
    tabs = "\n".join(f"- **{g}**: " + ", ".join(v) for g, v in groups.items())
    card = f"""---
license: other
language: [pt, en]
tags: [football, soccer, futebol, sports]
pretty_name: FutData
size_categories: [10M<n<100M]
configs:
{cfgs}
---
# FutData — base de dados de futebol

**{len(files)} tabelas.** Última atualização automática: {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC.

Reúne em Parquet as maiores fontes abertas e gratuitas de futebol — times, jogadores, partidas, gols, cartões, faltas,
lesões, escalações, eventos lance a lance, transferências e valores de mercado — e é atualizado todos os dias por
[GitHub Actions](https://github.com/alisson990jd/l/tree/main/futdata).

| Pasta | Fonte e licença | Cobertura | Conteúdo |
|---|---|---|---|
{tbl}

## Tabelas
{tabs}

## Uso
```python
from datasets import load_dataset
ds = load_dataset("Alisson990/futdata", "sportmonks__player_match_stats")
# ou com pandas
import pandas as pd
df = pd.read_parquet("hf://datasets/Alisson990/futdata/data/statsbomb/events.parquet")
```

O dicionário de colunas de `api_football/` está em `data/api_football/DATA_DICTIONARY.md`. Em `statsbomb/events`, a coluna
`details` traz o JSON completo do evento (xG do chute, tipo de passe, falta, cartão…).

As licenças seguem as fontes originais; cite-as ao usar.
"""
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/README.md", "w").write(card)
    print("README with", len(files), "tables")


if __name__ == "__main__":
    main()
