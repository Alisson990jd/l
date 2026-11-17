"""Sportmonks (free plan: Premier League, Bundesliga, Superliga DK, Premiership SCO).

Per match: events (goals, cards, subs), lineups with ~40 per-player stats (fouls, cards,
rating, passes, duels...), team statistics, injured/suspended players, referees, coaches.

usage: python -m futdata.sportmonks START END   (dates YYYY-MM-DD; default: last 10 days .. +7 days)
"""
import datetime as dt, os, sys, time, urllib.parse
import pandas as pd
from .common import http_json, merge_save, upload

B = "https://api.sportmonks.com/v3/football"
INC = ("participants;scores;state;league;season;round;venue;events.type;events.period;"
       "lineups.details.type;lineups.position;statistics.type;sidelined.type;sidelined.player;"
       "referees.referee;formations;coaches")


def fetch(start, end):
    token = os.environ["SPORTMONKS_TOKEN"]
    out, d = [], start
    while d <= end:
        e = min(d + dt.timedelta(days=29), end)
        q = urllib.parse.urlencode({"api_token": token, "include": INC, "per_page": 25})
        url = f"{B}/fixtures/between/{d}/{e}?{q}"
        while url:
            j, _ = http_json(url)
            out += j.get("data", [])
            p, rl = j.get("pagination") or {}, j.get("rate_limit") or {}
            url = f"{p['next_page']}&{q}" if p.get("has_more") else None
            if rl.get("remaining", 99) < 3:
                time.sleep(rl.get("resets_in_seconds", 3600) + 5)
        print(f"sportmonks {d}..{e}: {len(out)} fixtures so far", flush=True)
        d = e + dt.timedelta(days=1)
    return out


def flatten(fixtures):
    fx, ev, ps, ts, sl, rf = [], [], [], [], [], []
    for f in fixtures:
        teams = {p["meta"]["location"]: p for p in f.get("participants", [])}
        cur = {s["score"]["participant"]: s["score"]["goals"] for s in f.get("scores", []) if s.get("description") == "CURRENT"}
        ht = {s["score"]["participant"]: s["score"]["goals"] for s in f.get("scores", []) if s.get("description") == "1ST_HALF"}
        form = {x["location"]: x["formation"] for x in f.get("formations", [])}
        h, a = teams.get("home", {}), teams.get("away", {})
        fx.append({
            "fixture_id": f["id"], "date": f.get("starting_at"), "league_id": f.get("league_id"),
            "league": (f.get("league") or {}).get("name"), "season": (f.get("season") or {}).get("name"),
            "round": (f.get("round") or {}).get("name"), "state": (f.get("state") or {}).get("short_name"),
            "home_team_id": h.get("id"), "home_team": h.get("name"), "away_team_id": a.get("id"), "away_team": a.get("name"),
            "home_goals": cur.get("home"), "away_goals": cur.get("away"), "home_goals_ht": ht.get("home"), "away_goals_ht": ht.get("away"),
            "home_formation": form.get("home"), "away_formation": form.get("away"),
            "venue": (f.get("venue") or {}).get("name"), "result_info": f.get("result_info"),
            "referee": next((r["referee"]["name"] for r in f.get("referees", []) if r.get("type_id") == 6 and r.get("referee")), None),
            "coaches": "; ".join(c.get("name", "") for c in f.get("coaches", [])),
        })
        for e in f.get("events", []):
            ev.append({"fixture_id": f["id"], "event_id": e["id"], "type": (e.get("type") or {}).get("name"),
                       "minute": e.get("minute"), "extra_minute": e.get("extra_minute"), "team_id": e.get("participant_id"),
                       "player_id": e.get("player_id"), "player": e.get("player_name"),
                       "related_player_id": e.get("related_player_id"), "related_player": e.get("related_player_name"),
                       "result": e.get("result"), "info": e.get("info"), "addition": e.get("addition"), "injured": e.get("injured")})
        for l in f.get("lineups", []):
            row = {"fixture_id": f["id"], "player_id": l.get("player_id"), "player": l.get("player_name"), "team_id": l.get("team_id"),
                   "jersey_number": l.get("jersey_number"), "position": (l.get("position") or {}).get("name"),
                   "starter": l.get("type_id") == 11, "formation_position": l.get("formation_position")}
            for d in l.get("details", []):
                v = (d.get("data") or {}).get("value")
                row[(d.get("type") or {}).get("name", str(d.get("type_id")))] = v if not isinstance(v, dict) else str(v)
            ps.append(row)
        for s in f.get("statistics", []):
            ts.append({"fixture_id": f["id"], "team_id": s.get("participant_id"), "location": s.get("location"),
                       "stat": (s.get("type") or {}).get("name"), "value": (s.get("data") or {}).get("value")})
        for s in f.get("sidelined", []):
            sl.append({"fixture_id": f["id"], "date": f.get("starting_at"), "team_id": s.get("participant_id"),
                       "player_id": s.get("player_id"), "player": (s.get("player") or {}).get("name"),
                       "reason": (s.get("type") or {}).get("name"), "category": (s.get("type") or {}).get("model_type")})
    tsw = pd.DataFrame(ts)
    if not tsw.empty:
        tsw = tsw.pivot_table(index=["fixture_id", "team_id", "location"], columns="stat", values="value", aggfunc="first").reset_index()
    return {"fixtures": (pd.DataFrame(fx), ["fixture_id"]), "events": (pd.DataFrame(ev), ["event_id"]),
            "player_match_stats": (pd.DataFrame(ps), ["fixture_id", "player_id"]),
            "team_match_stats": (tsw, ["fixture_id", "team_id"]), "sidelined_injuries": (pd.DataFrame(sl), ["fixture_id", "player_id"])}


def main():
    today = dt.date.today()
    start = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else today - dt.timedelta(days=10)
    end = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else today + dt.timedelta(days=7)
    for name, (df, keys) in flatten(fetch(start, end)).items():
        merge_save(f"sportmonks/{name}", df, keys)


if __name__ == "__main__":
    main()
    if os.environ.get("FUTDATA_UPLOAD") == "1":
        upload("Sportmonks update")
