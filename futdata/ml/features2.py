"""Pre-match features from the API-Football tables. Everything uses only data strictly before kick-off."""
import sys, time, numpy as np, pandas as pd
from collections import deque, Counter
D=sys.argv[1]+"/data/api_football/"; OUT=sys.argv[2]; T0=time.time()
def log(*a): print(f"[{time.time()-T0:5.0f}s]",*a,flush=True)
f=pd.read_parquet(D+"fixtures.parquet",columns=["id","date_utc","league_id","home_team_id","away_team_id","goals_home","goals_away","is_played","referee_name"])
f=f[f.is_played&f.goals_home.notna()].rename(columns={"id":"fixture_id","date_utc":"date"}).copy()
f["date"]=pd.to_datetime(f.date)
# The source lists many matches twice under two league ids (same day, teams and score). A duplicate would leak the
# result into the second copy's form/Elo features, so keep one copy per match: the one with the most detail.
rich=sum(f.fixture_id.isin(pd.read_parquet(D+t,columns=["fixture_id"]).fixture_id).astype(int)*w
         for t,w in (("odds.parquet",4),("match_stats.parquet",2),("fixture_lineups.parquet",1)))
n0=len(f); f=f.assign(_r=rich,_day=f.date.dt.normalize()).sort_values(["_r","fixture_id"],ascending=[False,True])
f=f.drop_duplicates(["_day","home_team_id","away_team_id"]).drop(columns=["_r","_day"])
f=f.sort_values(["date","fixture_id"]).reset_index(drop=True); log("dedup",n0,"->",len(f))
f["hg"]=f.goals_home.astype(int); f["ag"]=f.goals_away.astype(int); f["result"]=np.select([f.hg>f.ag,f.hg==f.ag],[0,1],2)
odds=pd.read_parquet(D+"odds.parquet"); odds=odds.sort_values("bookmaker").drop_duplicates("fixture_id")  # prefer one bookmaker row (Pinnacle sorts after Bet365 -> keep first? fix below)
odds=pd.read_parquet(D+"odds.parquet"); odds["pri"]=(odds.bookmaker!="Pinnacle").astype(int); odds=odds.sort_values("pri").drop_duplicates("fixture_id")
inv=1/odds[["home_win","draw","away_win"]]; tot=inv.sum(1)
odds["oH"],odds["oD"],odds["oA"]=inv.home_win/tot,inv.draw/tot,inv.away_win/tot
f=f.merge(odds[["fixture_id","home_win","draw","away_win","oH","oD","oA","bookmaker"]],on="fixture_id",how="left")
# league-season id: new season when a league has a gap > 40 days
f["gap"]=f.groupby("league_id").date.diff().dt.days.fillna(999); f["season_id"]=(f.gap>40).groupby(f.league_id).cumsum()
f["ls"]=f.league_id.astype(str)+"_"+f.season_id.astype(str)
log("fixtures",len(f))

# ---- Elo
elo={}; eh=[]; ea=[]
for h,a,hg,ag in zip(f.home_team_id,f.away_team_id,f.hg,f.ag):
    rh,ra=elo.get(h,1500.),elo.get(a,1500.); eh.append(rh); ea.append(ra)
    e=1/(1+10**((ra-rh-60)/400)); s=1. if hg>ag else .5 if hg==ag else 0.; k=20*(1+abs(hg-ag))**.5
    elo[h]=rh+k*(s-e); elo[a]=ra-k*(s-e)
f["elo_h"],f["elo_a"]=eh,ea; f["elo_diff"]=f.elo_h-f.elo_a
log("elo")

# ---- team long table with match stats
ms=pd.read_parquet(D+"match_stats.parquet")
ST=["xg","shots_total","shots_on_goal","shots_inside_box","corners","possession","fouls","yellow_cards","red_cards","pass_accuracy","offsides"]
f=f.merge(ms[["fixture_id"]+[f"{s}_{x}" for x in ST for s in("home","away")]],on="fixture_id",how="left")
def side(s,o,flag):
    d=pd.DataFrame({"fixture_id":f.fixture_id,"date":f.date,"ls":f.ls,"team":f[f"{s}_team_id"],"home":flag,"gf":f[f"{s[0]}g"],"ga":f[f"{o[0]}g"]})
    for x in ST: d[f"{x}_for"]=f[f"{s}_{x}"]; d[f"{x}_ag"]=f[f"{o}_{x}"]
    return d
L=pd.concat([side("home","away",1),side("away","home",0)]).sort_values(["team","date","fixture_id"]).reset_index(drop=True)
L["pts"]=np.select([L.gf>L.ga,L.gf==L.ga],[3,1],0); L["gd"]=L.gf-L.ga
g=L.groupby("team",sort=False)
roll_cols=["gf","ga","pts"]+[c for c in L.columns if c.endswith(("_for","_ag"))]
sh=g[roll_cols].shift()
gs=sh.groupby(L.team)
for n in (5,10):
    r=gs.rolling(n,min_periods=1).mean().reset_index(level=0,drop=True)
    for c in roll_cols: L[f"{c}_r{n}"]=r[c]
ew=gs.ewm(halflife=8,ignore_na=True).mean().reset_index(level=0,drop=True)
for c in roll_cols: L[f"{c}_ew"]=ew[c]
L["rest"]=g.date.diff().dt.days.clip(upper=30)
dd=L.date.values.astype("datetime64[D]").astype(int)
L["games14"]=0
for t,idx in g.indices.items():
    x=dd[idx]; L.loc[L.index[idx],"games14"]=np.searchsorted(x,x,side="left")-np.searchsorted(x,x-14,side="left")
# standings inside league-season, before the match
gl=L.groupby(["team","ls"],sort=False)
L["ls_games"]=gl.cumcount(); L["ls_ppg"]=(gl.pts.cumsum()-L.pts)/L.ls_games.replace(0,np.nan); L["ls_gdpg"]=(gl.gd.cumsum()-L.gd)/L.ls_games.replace(0,np.nan)
L["home_ppg"]=(L.pts*L.home).groupby(L.team).transform(lambda s:s.shift().rolling(20,min_periods=1).sum())/L.home.groupby(L.team).transform(lambda s:s.shift().rolling(20,min_periods=1).sum()).replace(0,np.nan)
TEAM=[c for c in L.columns if c.endswith(("_r5","_r10","_ew"))]+["rest","games14","ls_games","ls_ppg","ls_gdpg","home_ppg"]
log("team features",len(TEAM))

# ---- players: per-player rolling form over last 10 played games (as-of before each fixture)
fp=pd.read_parquet(D+"fixture_players.parquet",columns=["id","fixture_id","team_id","player_id","is_starter","minutes","rating","position"])
fp=fp[fp.fixture_id.isin(f.fixture_id[f.date>="2013-07-01"])]
for c in ("minutes","rating"): fp[c]=fp[c].astype("float32")
st=pd.read_parquet(D+"fixture_players_stats_flat.parquet",columns=["fixture_player_id","goals_total","goals_assists","shots_total","shots_on","passes_key","passes_total","passes_accuracy","duels_total","duels_won","dribbles_success","tackles_total","tackles_interceptions","fouls_committed","fouls_drawn","cards_yellow","cards_red","goals_saves","goals_conceded"])
st=st[st.fixture_player_id.isin(fp.id)]
for c in st.columns:
    if c not in ("fixture_player_id","passes_accuracy"): st[c]=pd.to_numeric(st[c],errors="coerce").astype("float32")
fp=fp.merge(st,left_on="id",right_on="fixture_player_id",how="left").merge(f[["fixture_id","date"]],on="fixture_id")
fp["passes_accuracy"]=pd.to_numeric(fp.passes_accuracy,errors="coerce"); fp["acc_passes"]=fp.passes_total*fp.passes_accuracy/100
log("fixture_players",len(fp))
P=["minutes","goals_total","goals_assists","shots_total","shots_on","passes_key","passes_total","acc_passes","duels_total","duels_won","dribbles_success","tackles_total","tackles_interceptions","fouls_committed","fouls_drawn","cards_yellow","cards_red","goals_saves","goals_conceded"]
h=fp[fp.minutes.fillna(0)>0].copy()
for c in P: h[c]=h[c].fillna(0)
h["rated"]=h.rating.notna().astype(int); h["rsum"]=h.rating.fillna(0); h["starts"]=h.is_starter.astype(int); h["apps"]=1
h=h.sort_values(["player_id","date","fixture_id"]); hg_=h.groupby("player_id",sort=False)
cols=P+["rated","rsum","starts","apps"]
cum=hg_[cols].cumsum(); lag=cum.groupby(h.player_id).shift(10).fillna(0)
W=cum-lag  # sums over last 10 played games INCLUDING this row
hist=pd.DataFrame({"player_id":h.player_id.values,"date":h.date.values})
for c in cols: hist[f"w_{c}"]=W[c].values
hist["career_apps"]=cum.apps.values; hist["career_goals"]=cum.goals_total.values
hist=hist.sort_values("date")
xi=fp[fp.is_starter].sort_values("date"); bench=fp[~fp.is_starter].sort_values("date")
def asof(x):
    x=pd.merge_asof(x,hist,on="date",by="player_id",allow_exact_matches=False)
    m=x.w_minutes.clip(lower=90)/90
    x["p_rating"]=x.w_rsum/x.w_rated.replace(0,np.nan)
    for c in ["goals_total","goals_assists","shots_total","shots_on","passes_key","duels_won","dribbles_success","tackles_total","tackles_interceptions","fouls_committed","fouls_drawn","cards_yellow","goals_saves","goals_conceded"]:
        x[f"p_{c}90"]=x[f"w_{c}"]/m
    x["p_duel_pct"]=x.w_duels_won/x.w_duels_total.replace(0,np.nan); x["p_pass_pct"]=x.w_acc_passes/x.w_passes_total.replace(0,np.nan)
    x["p_min_share"]=x.w_minutes/(x.w_apps.replace(0,np.nan)*90); x["p_start_share"]=x.w_starts/x.w_apps.replace(0,np.nan)
    return x
xi=asof(xi); bench=asof(bench); log("player asof")
pc=[c for c in xi.columns if c.startswith("p_")]
agg={f"xi_{c}":(c,"mean") for c in pc}; agg.update({"xi_rating_max":("p_rating","max"),"xi_rating_min":("p_rating","min"),"xi_goals90_sum":("p_goals_total90","sum"),
     "xi_career_apps":("career_apps","mean"),"xi_career_goals":("career_goals","sum"),"xi_n":("player_id","size"),"xi_new":("career_apps",lambda s:s.isna().sum())})
X=xi.groupby(["fixture_id","team_id"]).agg(**agg).reset_index()
gk=xi[xi.position.eq("G")].groupby(["fixture_id","team_id"]).agg(gk_rating=("p_rating","max"),gk_saves90=("p_goals_saves90","max"),gk_conc90=("p_goals_conceded90","max")).reset_index()
B=bench.groupby(["fixture_id","team_id"]).agg(bench_rating=("p_rating","mean"),bench_rating_max=("p_rating","max"),bench_goals90=("p_goals_total90","sum")).reset_index()
X=X.merge(gk,how="left").merge(B,how="left")
# ---- absences / rotation: regulars of the last 5 matches missing today
xs=xi.groupby(["fixture_id","team_id"]).agg(pl=("player_id",list),rt=("p_rating",list)).reset_index().merge(f[["fixture_id","date"]]).sort_values("date")
out=[]; mem={}
for fid,tid,pl,rt in zip(xs.fixture_id,xs.team_id,xs.pl,xs.rt):
    q=mem.setdefault(tid,deque(maxlen=5)); cnt=Counter(p for lst,_ in q for p in lst)
    rate={}; [rate.update({p:r for p,r in zip(lst,rl) if r==r}) for lst,rl in q]
    reg={p for p,c in cnt.items() if c>=3}; today=set(pl); miss=reg-today
    last=set(q[-1][0]) if q else set()
    out.append((fid,tid,len(reg),len(miss),sum(rate.get(p,6.5) for p in miss),len(today&last) if q else np.nan,len(q)))
    q.append((pl,rt))
A=pd.DataFrame(out,columns=["fixture_id","team_id","abs_regulars","abs_missing","abs_missing_rating","abs_same_as_last","abs_hist"])
A.loc[A.abs_hist<3,["abs_missing","abs_missing_rating","abs_same_as_last"]]=np.nan
X=X.merge(A.drop(columns="abs_hist"),how="left")
PLAYER=[c for c in X.columns if c not in("fixture_id","team_id")]
import gc; del fp,h,hist,cum,lag,W,xi,bench,st,xs,sh,gs,ew; gc.collect()
log("player features",len(PLAYER))

# ---- league priors (prior 300 matches of the league)
lg=f.groupby("league_id")
for nm,s in (("lg_home",(f.result==0).astype(float)),("lg_draw",(f.result==1).astype(float)),("lg_goals",(f.hg+f.ag).astype(float))):
    f[nm]=s.groupby(f.league_id).transform(lambda x:x.shift().rolling(300,min_periods=20).mean())
f["ls_progress"]=f.groupby("ls").cumcount()/f.groupby("ls").fixture_id.transform("size")
# referee: prior cards per match
f["cards"]=f.home_yellow_cards+f.away_yellow_cards
f["ref_cards"]=f.groupby("referee_name").cards.transform(lambda x:x.shift().rolling(30,min_periods=3).mean())
# ---- assemble wide (only target rows)
f=f[f.date>="2014-07-01"]
lg_odds=f[f.oH.notna()].league_id.value_counts(); f=f[f.league_id.isin(lg_odds[lg_odds>=200].index)].copy()
L=L[L.fixture_id.isin(f.fixture_id)]; X=X[X.fixture_id.isin(f.fixture_id)]
for c in TEAM: L[c]=L[c].astype("float32")
for c in PLAYER: X[c]=X[c].astype("float32")
gc.collect()
for s,flag in (("h",1),("a",0)):
    t=L[L.home==flag][["fixture_id"]+TEAM].rename(columns={c:f"{c}_{s}" for c in TEAM}); f=f.merge(t,on="fixture_id",how="left")
    p=X.rename(columns={"team_id":("home" if s=="h" else "away")+"_team_id",**{c:f"{c}_{s}" for c in PLAYER}}); f=f.merge(p,on=["fixture_id",("home" if s=="h" else "away")+"_team_id"],how="left")
f=pd.concat([f,pd.DataFrame({f"{c}_d":f[f"{c}_h"]-f[f"{c}_a"] for c in TEAM+PLAYER},index=f.index)],axis=1)
f.drop(columns=[c for c in f.columns if c.startswith(("home_","away_")) and not c.endswith(("_h","_a","_d")) and c not in("home_team_id","away_team_id","home_win","away_win")]+["gap","goals_home","goals_away","is_played"],errors="ignore").to_parquet(OUT,index=False)
import json; json.dump({"TEAM":TEAM,"PLAYER":PLAYER},open(OUT+".cols.json","w"))
log("saved",f.shape, "with odds",f.oH.notna().sum(),"with xi",f.xi_n_h.notna().sum())
