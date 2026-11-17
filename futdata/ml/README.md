# Experimento de previsão (XGBoost)

Para cada jogo do Transfermarkt (88.768 jogos, 2012–2026), usa apenas informação anterior ao apito inicial:
Elo dos times, forma recente, valor de mercado dos titulares e do banco na data do jogo, idade, altura,
gols/assistências/cartões dos titulares nos 10 jogos anteriores e na carreira.

Validação walk-forward: treina com todas as temporadas anteriores e prevê a temporada inteira seguinte (2022, 2023, 2024, 2025).

```bash
python futdata/ml/features.py <pasta com data/> matches.parquet
python futdata/ml/model.py matches.parquet .
python futdata/ml/bookies.py <scratch>   # compara com as odds de fechamento do football-data.co.uk
```
