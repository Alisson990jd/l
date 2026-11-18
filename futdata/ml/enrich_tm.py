"""Link API-Football fixtures to Transfermarkt games and add lineup market-value features."""
import sys, re, unicodedata, difflib, numpy as np, pandas as pd
from collections import Counter, defaultdict
S=sys.argv[1]
m2=pd.read_parquet(f"{S}/ml/m2.parquet")
teams=pd.read_parquet(f"{S}/out/data/api_football/teams.parquet",columns=["id","name"]).set_index("id").name
tm=pd.read_parquet(f"{S}/ml/matches.parquet")
tmf=[c for c in tm.columns if c.startswith(("xi_","gk_","bench_"))]
tm=tm[["game_id","date","home_club_id","away_club_id","home_club_name","away_club_name","hg","ag"]+tmf]
def norm(s):
    s=unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode().lower()
    s=re.sub(r"\b(football club|futebol clube|fc|cf|ac|sc|sv|as|afc|club|de|calcio|the|cd|ud|sd|rc|fk|sk|if|bk|ss|ssc|vfl|vfb|tsg|1\.)\b"," ",s)
    return re.sub(r"[^a-z0-9 ]"," ",s).split()
def sim(a,b):
    a,b=" ".join(norm(a))," ".join(norm(b)); return difflib.SequenceMatcher(None,a,b).ratio()
a=m2[["fixture_id","date","home_team_id","away_team_id","hg","ag"]].copy(); a["d"]=a.date.dt.normalize()
tm["d"]=pd.to_datetime(tm.date).dt.normalize()
cands=pd.concat([a.merge(tm.assign(d=tm.d+pd.Timedelta(days=o)),on=["d","hg","ag"]) for o in (-1,0,1)])
print("candidate pairs",len(cands))
votes=defaultdict(Counter)
cache={}
def s2(x,y):
    k=(x,y)
    if k not in cache: cache[k]=sim(teams.get(x,""),y)
    return cache[k]
for r in cands.itertuples():
    sh,sa=s2(r.home_team_id,r.home_club_name),s2(r.away_team_id,r.away_club_name)
    if sh+sa>1.0:
        votes[r.home_team_id][r.home_club_id]+=1; votes[r.away_team_id][r.away_club_id]+=1
mp={}
for t,c in votes.items():
    (club,n),tot=c.most_common(1)[0],sum(c.values())
    if n>=3 and n/tot>=0.6: mp[t]=club
print("api teams mapped",len(mp),"of",len(set(a.home_team_id)|set(a.away_team_id)))
a["home_club_id"]=a.home_team_id.map(mp); a["away_club_id"]=a.away_team_id.map(mp)
j=pd.concat([a.dropna(subset=["home_club_id"]).merge(tm.assign(d=tm.d+pd.Timedelta(days=o)).drop(columns=["date","hg","ag"]),on=["d","home_club_id","away_club_id"]) for o in (0,-1,1)])
j=j.drop_duplicates("fixture_id")
print("fixtures linked",len(j),"of",len(m2),"| with odds:", m2[m2.fixture_id.isin(j.fixture_id)].oH.notna().sum())
out=m2.merge(j[["fixture_id"]+tmf].rename(columns={c:"tm_"+c for c in tmf}),on="fixture_id",how="left")
out.to_parquet(f"{S}/ml/m3.parquet",index=False)
import json; C=json.load(open(f"{S}/ml/m2.parquet.cols.json")); C["TM"]=["tm_"+c for c in tmf]; json.dump(C,open(f"{S}/ml/m3.parquet.cols.json","w"))
print("saved",out.shape)
