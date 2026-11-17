"""Refresh the free open sources that keep publishing new data.

- football-data.co.uk: current season, 22 European divisions (match stats: fouls, cards, shots, corners, odds) + 16 extra leagues
- martj42/international_results: every national-team match, goalscorers, shootouts
- transfermarkt-datasets (dcaribou): clubs, players, games, events, lineups, appearances, valuations, transfers
"""
import datetime as dt, gzip, io, os
import pandas as pd
from .common import http_bytes, existing, merge_save, upload

DIVS = "E0 E1 E2 E3 EC SC0 SC1 SC2 SC3 D1 D2 I1 I2 SP1 SP2 F1 F2 N1 B1 P1 T1 G1".split()
EXTRA = "ARG AUT BRA CHN DNK FIN IRL JPN MEX NOR POL ROU RUS SWE SWZ USA".split()
TM = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
TM_TABLES = ["competitions", "clubs", "players", "games", "club_games", "appearances", "game_events", "game_lineups", "player_valuations", "transfers"]


def season_code(today):
    y = today.year if today.month >= 7 else today.year - 1
    return f"{y % 100:02d}{(y + 1) % 100:02d}", f"{y}-{(y + 1) % 100:02d}"


def football_data():
    code, label = season_code(dt.date.today())
    parts = []
    for d in DIVS:
        try:
            df = pd.read_csv(io.BytesIO(http_bytes(f"https://www.football-data.co.uk/mmz4281/{code}/{d}.csv")), encoding="latin-1", dtype=str)
            df = df.dropna(how="all").dropna(axis=1, how="all")
            parts.append(df.loc[:, ~df.columns.str.startswith("Unnamed")])
        except Exception as e:
            print("skip", d, e)
    if parts:
        new = pd.concat(parts, ignore_index=True)
        new.insert(0, "Season", label)
        old = existing("football_data_uk/main_leagues")
        if not old.empty:
            old = old[old["Season"] != label]
        merge_save("football_data_uk/main_leagues", pd.concat([old, new], ignore_index=True).astype("string"), ["Season", "Div", "Date", "HomeTeam", "AwayTeam"], merge=False)
    extra = []
    for c in EXTRA:
        try:
            extra.append(pd.read_csv(io.BytesIO(http_bytes(f"https://www.football-data.co.uk/new/{c}.csv")), encoding="latin-1", dtype=str))
        except Exception as e:
            print("skip", c, e)
    if extra:
        merge_save("football_data_uk/extra_leagues", pd.concat(extra, ignore_index=True), ["Country", "League", "Season", "Date", "Home", "Away"], merge=False)


def international():
    base = "https://raw.githubusercontent.com/martj42/international_results/master/"
    for t, keys in [("results", ["date", "home_team", "away_team"]), ("goalscorers", ["date", "home_team", "away_team", "scorer", "minute"]),
                    ("shootouts", ["date", "home_team", "away_team"]), ("former_names", ["current", "former"])]:
        merge_save(f"international/{t}", pd.read_csv(io.BytesIO(http_bytes(base + t + ".csv"))), keys, merge=False)


def transfermarkt():
    keys = {"competitions": ["competition_id"], "clubs": ["club_id"], "players": ["player_id"], "games": ["game_id"],
            "club_games": ["game_id", "club_id"], "appearances": ["appearance_id"], "game_events": ["game_event_id"],
            "game_lineups": ["game_lineups_id"], "player_valuations": ["player_id", "date"], "transfers": ["player_id", "transfer_date", "from_club_id", "to_club_id"]}
    for t in TM_TABLES:
        df = pd.read_csv(io.BytesIO(gzip.decompress(http_bytes(TM + t + ".csv.gz"))), low_memory=False)
        merge_save(f"transfermarkt/{t}", df, keys[t], merge=False)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["football_data", "international", "transfermarkt"]
    for w in which:
        try:
            globals()[w]()
        except Exception as e:
            print(f"{w} failed: {e}", flush=True)
    if os.environ.get("FUTDATA_UPLOAD") == "1":
        upload("Open sources refresh")
