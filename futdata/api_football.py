"""API-Football daily collector (works on the free plan: 100 requests/day, dates yesterday..tomorrow).

1 request per date gives every fixture of the day worldwide (~1,100 leagues) with score.
1 request per finished fixture gives events, lineups, team stats and per-player stats
(fouls, cards, minutes, rating, shots, passes...). Remaining quota goes to details,
priority leagues first. 1 request for the day's injuries.

usage: python -m futdata.api_football [DATE ...]   (default: yesterday and today)
"""
import datetime as dt, os, sys, time
import pandas as pd
from .common import http_json, merge_save, upload

A = "https://v3.football.api-sports.io"
# Priority for per-match detail: leagues NOT covered by Sportmonks free (EPL/Bundesliga are there).
PRIORITY = [71, 72, 73, 13, 11, 2, 3, 848, 140, 135, 61, 94, 88, 128, 253, 262, 307, 1, 4, 5, 9, 10, 29, 30, 31, 32, 34,
            75, 76, 141, 136, 62, 79, 144, 203, 197, 119, 113, 103, 106, 218, 207, 239, 265, 268, 281, 242, 98, 292, 169, 188]
DONE = {"FT", "AET", "PEN"}
state = {"left": 100}


def call(path):
    if state["left"] <= 1:
        raise StopIteration
    j, h = http_json(A + path, {"x-apisports-key": os.environ["APIFOOTBALL_KEY"]})
    state["left"] = int(h.get("x-ratelimit-requests-remaining", state["left"] - 1))
    if j.get("errors"):
        print("api errors:", j["errors"], path, flush=True)
    if int(h.get("x-ratelimit-remaining", 10)) <= 1:
        time.sleep(61)  # 10 requests/minute on free plan
    return j.get("response", [])


def fx_row(f):
    fi, lg, t, g, s = f["fixture"], f["league"], f["teams"], f["goals"], f["score"]
    return {"fixture_id": fi["id"], "date": fi["date"], "status": fi["status"]["short"], "elapsed": fi["status"].get("elapsed"),
            "referee": fi.get("referee"), "venue": (fi.get("venue") or {}).get("name"), "city": (fi.get("venue") or {}).get("city"),
            "league_id": lg["id"], "league": lg["name"], "country": lg["country"], "season": lg["season"], "round": lg.get("round"),
            "home_team_id": t["home"]["id"], "home_team": t["home"]["name"], "away_team_id": t["away"]["id"], "away_team": t["away"]["name"],
            "home_goals": g["home"], "away_goals": g["away"], "home_goals_ht": s["halftime"]["home"], "away_goals_ht": s["halftime"]["away"],
            "home_goals_et": s["extratime"]["home"], "away_goals_et": s["extratime"]["away"], "home_pen": s["penalty"]["home"], "away_pen": s["penalty"]["away"]}


def main():
    today = dt.date.today()
    dates = sys.argv[1:] or [str(today - dt.timedelta(days=1)), str(today)]
    fixtures, events, players, tstats, lineups, injuries = [], [], [], [], [], []
    try:
        j = http_json(A + "/status", {"x-apisports-key": os.environ["APIFOOTBALL_KEY"]})[0]
        if j.get("errors") or not isinstance(j.get("response"), dict):
            print("api-football unavailable:", j.get("errors"), flush=True)
            return
        st = j["response"]["requests"]
        state["left"] = st["limit_day"] - st["current"]
        print("api-football requests left today:", state["left"], flush=True)
        day = []
        for d in dates:
            day += call(f"/fixtures?date={d}")
        fixtures = [fx_row(f) for f in day]
        try:
            for i in call(f"/injuries?date={dates[-1]}"):
                injuries.append({"fixture_id": (i.get("fixture") or {}).get("id"), "date": (i.get("fixture") or {}).get("date"),
                                 "league_id": i["league"]["id"], "team_id": i["team"]["id"], "team": i["team"]["name"],
                                 "player_id": i["player"]["id"], "player": i["player"]["name"],
                                 "type": i["player"].get("type"), "reason": i["player"].get("reason")})
        except StopIteration:
            pass
        rank = {l: k for k, l in enumerate(PRIORITY)}
        todo = sorted([f for f in day if f["fixture"]["status"]["short"] in DONE], key=lambda f: rank.get(f["league"]["id"], 999))
        for f in todo:
            fid = f["fixture"]["id"]
            r = call(f"/fixtures?id={fid}")
            if not r:
                continue
            r = r[0]
            for e in r.get("events", []):
                events.append({"fixture_id": fid, "minute": e["time"]["elapsed"], "extra": e["time"].get("extra"), "team_id": e["team"]["id"],
                               "team": e["team"]["name"], "player_id": (e.get("player") or {}).get("id"), "player": (e.get("player") or {}).get("name"),
                               "assist_id": (e.get("assist") or {}).get("id"), "assist": (e.get("assist") or {}).get("name"),
                               "type": e.get("type"), "detail": e.get("detail"), "comments": e.get("comments")})
            for t in r.get("statistics", []):
                row = {"fixture_id": fid, "team_id": t["team"]["id"], "team": t["team"]["name"]}
                row.update({s["type"]: s["value"] for s in t.get("statistics", [])})
                tstats.append(row)
            for l in r.get("lineups", []):
                for kind in ("startXI", "substitutes"):
                    for p in l.get(kind) or []:
                        p = p["player"]
                        lineups.append({"fixture_id": fid, "team_id": l["team"]["id"], "formation": l.get("formation"),
                                        "coach": (l.get("coach") or {}).get("name"), "player_id": p.get("id"), "player": p.get("name"),
                                        "number": p.get("number"), "pos": p.get("pos"), "grid": p.get("grid"), "starter": kind == "startXI"})
            for t in r.get("players", []):
                for p in t.get("players", []):
                    s = (p.get("statistics") or [{}])[0]
                    row = {"fixture_id": fid, "team_id": t["team"]["id"], "player_id": p["player"]["id"], "player": p["player"]["name"]}
                    for grp, vals in s.items():
                        if isinstance(vals, dict):
                            row.update({f"{grp}_{k}": v for k, v in vals.items()})
                    players.append(row)
    except StopIteration:
        print("daily quota used up", flush=True)
    merge_save("api_football_live/fixtures", pd.DataFrame(fixtures), ["fixture_id"])
    merge_save("api_football_live/events", pd.DataFrame(events), ["fixture_id", "minute", "extra", "team_id", "player_id", "type", "detail"])
    merge_save("api_football_live/team_match_stats", pd.DataFrame(tstats), ["fixture_id", "team_id"])
    merge_save("api_football_live/lineups", pd.DataFrame(lineups), ["fixture_id", "player_id"])
    merge_save("api_football_live/player_match_stats", pd.DataFrame(players), ["fixture_id", "player_id"])
    merge_save("api_football_live/injuries", pd.DataFrame(injuries), ["fixture_id", "player_id", "date"])


if __name__ == "__main__":
    main()
    if os.environ.get("FUTDATA_UPLOAD") == "1":
        upload("API-Football daily update")
