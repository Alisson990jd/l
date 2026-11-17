"""Build one row per match with ONLY information known before kick-off."""
import sys, numpy as np, pandas as pd
D=sys.argv[1]+"/data/transfermarkt/"; OUT=sys.argv[2]
g=pd.read_parquet(D+"games.parquet")
g=g[(g.season>=2012)&g.home_club_goals.notna()&g.away_club_goals.notna()].copy()
g["date"]=pd.to_datetime(g.date); g=g.sort_values(["date","game_id"]).reset_index(drop=True)
g["hg"]=g.home_club_goals.astype(int); g["ag"]=g.away_club_goals.astype(int)
g["result"]=np.select([g.hg>g.ag,g.hg==g.ag],[0,1],2)   # 0 home win, 1 draw, 2 away win

# 1) Elo (pre-match), goal-difference weighted, home advantage 60
elo={}; eh=[]; ea=[]
for h,a,hg,ag in zip(g.home_club_id,g.away_club_id,g.hg,g.ag):
    rh,ra=elo.get(h,1500.),elo.get(a,1500.); eh.append(rh); ea.append(ra)
    exp=1/(1+10**((ra-rh-60)/400)); s=1. if hg>ag else .5 if hg==ag else 0.
    k=20*(1+abs(hg-ag))**0.5; elo[h]=rh+k*(s-exp); elo[a]=ra-k*(s-exp)
g["elo_h"],g["elo_a"]=eh,ea; g["elo_diff"]=g.elo_h-g.elo_a

# 2) Team form (shifted rolling: only previous matches)
long=pd.concat([pd.DataFrame({"game_id":g.game_id,"date":g.date,"team":g.home_club_id,"gf":g.hg,"ga":g.ag,"home":1}),
                pd.DataFrame({"game_id":g.game_id,"date":g.date,"team":g.away_club_id,"gf":g.ag,"ga":g.hg,"home":0})]).sort_values(["team","date","game_id"])
long["pts"]=np.select([long.gf>long.ga,long.gf==long.ga],[3,1],0)
grp=long.groupby("team")
for n in (5,10):
    for c in ("gf","ga","pts"):
        long[f"{c}_l{n}"]=grp[c].transform(lambda s: s.shift().rolling(n,min_periods=1).mean())
long["rest_days"]=grp.date.diff().dt.days.clip(upper=30)
long["n_prev"]=grp.cumcount()
tf=[c for c in long.columns if c.endswith(("_l5","_l10"))]+["rest_days","n_prev"]
for side,flag in (("h",1),("a",0)):
    x=long[long.home==flag][["game_id"]+tf].rename(columns={c:f"{c}_{side}" for c in tf}); g=g.merge(x,on="game_id",how="left")

# 3) Starting XI and bench: market value at match date (as-of last valuation BEFORE the match)
lu=pd.read_parquet(D+"game_lineups.parquet",columns=["game_id","player_id","club_id","type","position","team_captain"])
lu=lu[lu.game_id.isin(g.game_id)].merge(g[["game_id","date"]],on="game_id")
pv=pd.read_parquet(D+"player_valuations.parquet",columns=["player_id","date","market_value_in_eur"]); pv["date"]=pd.to_datetime(pv.date)
lu=lu.sort_values("date"); pv=pv.sort_values("date")
lu=pd.merge_asof(lu,pv.rename(columns={"market_value_in_eur":"mv"}),on="date",by="player_id",allow_exact_matches=False)
lu["lmv"]=np.log1p(lu.mv)

# 4) Player form from appearances (strictly before match): career + last 10 games
ap=pd.read_parquet(D+"appearances.parquet",columns=["player_id","game_id","date","goals","assists","minutes_played","yellow_cards","red_cards"])
ap["date"]=pd.to_datetime(ap.date); ap=ap.sort_values(["player_id","date","game_id"])
pg=ap.groupby("player_id")
for c in ("goals","assists","minutes_played","yellow_cards","red_cards"):
    ap[f"{c}_cum"]=pg[c].cumsum()-ap[c]
    ap[f"{c}_r10"]=pg[c].transform(lambda s: s.shift().rolling(10,min_periods=1).sum())
ap["apps_cum"]=pg.cumcount()
ap["days_since_last"]=pg.date.diff().dt.days
pf=ap[["player_id","game_id"]+[c for c in ap.columns if c.endswith(("_cum","_r10"))]+["days_since_last"]]
# features for match M = state after previous appearance + that appearance itself; use the row of the match itself (cum excludes it)
lu=lu.merge(pf,on=["player_id","game_id"],how="left")
pl=pd.read_parquet(D+"players.parquet",columns=["player_id","date_of_birth","height_in_cm"]); pl["dob"]=pd.to_datetime(pl.date_of_birth,errors="coerce")
lu=lu.merge(pl[["player_id","dob","height_in_cm"]],on="player_id",how="left"); lu["age"]=(lu.date-lu.dob).dt.days/365.25
lu["is_gk"]=lu.position.eq("Goalkeeper"); lu["is_fw"]=lu.position.str.contains("Forward|Striker|Winger",na=False)
lu["g90_r10"]=lu.goals_r10/lu.minutes_played_r10.clip(lower=90)*90
lu["ga90_r10"]=(lu.goals_r10+lu.assists_r10)/lu.minutes_played_r10.clip(lower=90)*90
lu["card90_cum"]=(lu.yellow_cards_cum+3*lu.red_cards_cum)/lu.minutes_played_cum.clip(lower=90)*90
st=lu[lu.type=="starting_lineup"]; sb=lu[lu.type=="substitutes"]
agg=st.groupby(["game_id","club_id"]).agg(xi_n=("player_id","size"),xi_mv_sum=("mv","sum"),xi_lmv_mean=("lmv","mean"),xi_lmv_max=("lmv","max"),xi_lmv_min=("lmv","min"),
    xi_mv_missing=("mv",lambda s:s.isna().mean()),xi_age=("age","mean"),xi_height=("height_in_cm","mean"),xi_apps=("apps_cum","mean"),
    xi_goals_r10=("goals_r10","sum"),xi_ga90=("ga90_r10","mean"),xi_fw_g90=("g90_r10","max"),xi_min_r10=("minutes_played_r10","mean"),
    xi_cards90=("card90_cum","mean"),xi_rest=("days_since_last","median"),xi_career_goals=("goals_cum","sum"),xi_fw_n=("is_fw","sum")).reset_index()
agg["xi_lmv_sum"]=np.log1p(agg.xi_mv_sum)
gk=st[st.is_gk].groupby(["game_id","club_id"]).agg(gk_lmv=("lmv","max"),gk_apps=("apps_cum","max")).reset_index()
bn=sb.groupby(["game_id","club_id"]).agg(bench_lmv_mean=("lmv","mean"),bench_lmv_max=("lmv","max")).reset_index()
agg=agg.merge(gk,how="left").merge(bn,how="left").drop(columns="xi_mv_sum")
for side,col in (("h","home_club_id"),("a","away_club_id")):
    x=agg.rename(columns={"club_id":col,**{c:f"{c}_{side}" for c in agg.columns if c not in("game_id","club_id")}}); g=g.merge(x,on=["game_id",col],how="left")
for c in [c[:-2] for c in g.columns if c.endswith("_h") and c[:-2]+"_a" in g.columns]:
    g[f"{c}_diff"]=g[f"{c}_h"]-g[f"{c}_a"]
g["comp_type"]=g.competition_type.astype("category").cat.codes
g.to_parquet(OUT,index=False); print(g.shape, "with lineups:",g.xi_n_h.notna().mean().round(3))
