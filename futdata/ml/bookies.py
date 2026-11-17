import sys, difflib, re, numpy as np, pandas as pd
from sklearn.metrics import log_loss
S=sys.argv[1]
p=pd.read_parquet(f"{S}/ml/preds.parquet"); p["date"]=pd.to_datetime(p.date).dt.normalize()
fd=pd.read_parquet(f"{S}/out/data/football_data_uk/main_leagues.parquet")
fd["date"]=pd.to_datetime(fd.Date,dayfirst=True,errors="coerce",format="mixed")
MAP={"GB1":"E0","ES1":"SP1","IT1":"I1","L1":"D1","FR1":"F1","NL1":"N1","PO1":"P1","BE1":"B1","TR1":"T1","GR1":"G1","SC1":"SC0"}
fd=fd[fd.Div.isin(MAP.values())]; p=p[p.competition_id.isin(MAP)].copy(); p["Div"]=p.competition_id.map(MAP)
def norm(s): return re.sub(r"\b(fc|cf|ac|sc|sv|ssc|as|afc|club|football|futbol|de|calcio|1\.|vfb|vfl|tsg|rcd|ud|sd|cd|rc|ogc|stade|sporting|associazione|sportiva|clube|real)\b","",str(s).lower()).strip()
fdg={k:v for k,v in fd.groupby(["Div","date"])}
rows=[]
for (div,d),grp in p.groupby(["Div","date"]):
    cand=pd.concat([fdg.get((div,d+pd.Timedelta(days=o)),pd.DataFrame()) for o in (0,-1,1)])
    if cand.empty: continue
    for _,r in grp.iterrows():
        sc=[(difflib.SequenceMatcher(None,norm(r.home_club_name),norm(c.HomeTeam)).ratio()+difflib.SequenceMatcher(None,norm(r.away_club_name),norm(c.AwayTeam)).ratio(),i) for i,c in cand.iterrows()]
        best,i=max(sc)
        c=cand.loc[i]
        if best>0.9 and int(c.FTHG)==r.hg and int(c.FTAG)==r.ag: rows.append({**r.to_dict(),**{k:c.get(k) for k in ["PSCH","PSCD","PSCA","B365CH","B365CD","B365CA","B365H","B365D","B365A","MaxCH","MaxCD","MaxCA","AvgCH","AvgCD","AvgCA"]}})
m=pd.DataFrame(rows); print("matched",len(m),"of",len(p))
for c in m.columns:
    if c[:3] in("PSC","B36","Max","Avg"): m[c]=pd.to_numeric(m[c],errors="coerce")
# closing odds: Pinnacle, fallback Bet365 closing, fallback Bet365
for o in "HDA":
    m[f"cl{o}"]=m[f"PSC{o}"].fillna(m[f"B365C{o}"]).fillna(m[f"B365{o}"])
m=m.dropna(subset=["clH","clD","clA"])
inv=1/m[["clH","clD","clA"]].values; book=inv/inv.sum(1,keepdims=True)
y=m.result.values
def rps(p,y):
    o=np.eye(3)[y]; return np.mean(np.sum((np.cumsum(p,1)-np.cumsum(o,1))[:,:2]**2,1)/2)
models={"Casa de aposta (odds de fechamento)":book,"XGBoost times+escalação":m[["pH","pD","pA"]].values,"XGBoost só times":m[["pH_team","pD_team","pA_team"]].values}
models["Média modelo+casa"]=(book+models["XGBoost times+escalação"])/2
out=[]
for k,v in models.items(): out.append({"modelo":k,"jogos":len(m),"acerto":(v.argmax(1)==y).mean(),"logloss":log_loss(y,v,labels=[0,1,2]),"rps":rps(v,y)})
print(pd.DataFrame(out).round(4).to_string(index=False))
# betting simulation: flat 1 unit on outcomes where model prob*best odds > 1+edge, using MAX closing odds across bookmakers (fallback closing)
mp=models["XGBoost times+escalação"]
for src in ("cl","MaxC"):
    od=np.column_stack([m[f"{src}{o}"].fillna(m[f"cl{o}"]) for o in "HDA"])
    for edge in (0.0,0.05,0.10,0.20):
        ev=mp*od-1; bet=ev>edge; profit=np.where(np.eye(3)[y].astype(bool)&bet,od,0)-bet
        print(f"odds={src:5s} edge>{edge:.2f}: apostas={bet.sum():6d}  lucro={profit.sum():8.1f}  ROI={profit.sum()/max(bet.sum(),1):+.2%}")
m.to_parquet(f"{S}/ml/matched_bookies.parquet",index=False)
