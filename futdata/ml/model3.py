import os, sys, json, time, numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb, optuna
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
optuna.logging.set_verbosity(optuna.logging.WARNING)
T0=time.time(); log=lambda *a: print(f"[{time.time()-T0:5.0f}s]",*a,flush=True)
g=pd.read_parquet(sys.argv[1]); C=json.load(open(sys.argv[1]+".cols.json")); OUTD=sys.argv[2]; os.makedirs(OUTD,exist_ok=True)
DEVICE=os.environ.get("XGB_DEVICE","cpu"); NJ=int(os.environ.get("N_JOBS","4"))
g["year"]=g.date.dt.year; y=g.result.values
_dup=g.assign(_d=g.date.dt.normalize()).duplicated(["_d","home_team_id","away_team_id"]).sum()
assert _dup==0, f"{_dup} duplicated matches (same day and teams): they leak results into features, rebuild features2"
hd=lambda cs:[f"{c}_{s}" for c in cs for s in("h","a","d")]
base_team=[c for c in C["TEAM"] if c.split("_r")[0] in("gf","ga","pts") or c.split("_ew")[0] in("gf","ga","pts")]
stats_team=[c for c in C["TEAM"] if (c.endswith(("_r5","_r10","_ew"))) and c not in base_team]
ctx_team=["rest","games14","ls_games","ls_ppg","ls_gdpg","home_ppg"]
player=[c for c in C["PLAYER"] if not c.startswith("abs_")]; absn=[c for c in C["PLAYER"] if c.startswith("abs_")]
S0=["elo_h","elo_a","elo_diff"]+hd(base_team)
S1=S0+hd(stats_team)
S2=S1+hd(player)
S3=S2+hd(absn)
S4=S3+hd(ctx_team)+["lg_home","lg_draw","lg_goals","ls_progress","ref_cards"]
TMF=C.get("TM",[])
S5=S4+TMF
SETS={"1. Elo + forma (gols/pontos)":S0,"2. + xG e estatísticas do time":S1,"3. + nota/estatísticas dos titulares":S2,"4. + desfalques e rodízio":S3,"5. + tabela, calendário, liga, árbitro":S4,"6. + valor de mercado dos titulares (Transfermarkt)":S5}
ODDS=["oH","oD","oA"]
def rps(p,y):
    o=np.eye(3)[y]; return np.mean(np.sum((np.cumsum(p,1)-np.cumsum(o,1))[:,:2]**2,1)/2)
def ll(p,y): return log_loss(y,np.clip(p,1e-6,1),labels=[0,1,2])
XP=dict(n_estimators=3000,learning_rate=0.03,max_depth=5,subsample=0.8,colsample_bytree=0.5,min_child_weight=30,reg_lambda=5,max_bin=256,tree_method="hist",device=DEVICE,n_jobs=NJ,early_stopping_rounds=150)
def fit_xgb(tr,va,fs,params=None,w=None,obj="multi:softprob",target="result"):
    p={**XP,**(params or {})}
    m=xgb.XGBClassifier(objective=obj,eval_metric="mlogloss",**p) if obj.startswith("multi") else xgb.XGBRegressor(objective=obj,**p)
    m.fit(tr[fs],tr[target],sample_weight=w,eval_set=[(va[fs],va[target])],verbose=False); return m
def dc_probs(lh,la,rho,maxg=10):
    i=np.arange(maxg+1); ph=poisson.pmf(i[None,:],lh[:,None]); pa=poisson.pmf(i[None,:],la[:,None])
    M=ph[:,:,None]*pa[:,None,:]
    M[:,0,0]*=1-lh*la*rho; M[:,0,1]*=1+lh*rho; M[:,1,0]*=1+la*rho; M[:,1,1]*=1-rho
    H=np.tril(np.ones((maxg+1,maxg+1)),-1); Dg=np.eye(maxg+1)
    out=np.stack([(M*H).sum((1,2)),(M*Dg).sum((1,2)),(M*H.T).sum((1,2))],1); return out/out.sum(1,keepdims=True)
def temp(p,T): q=p**(1/T); return q/q.sum(1,keepdims=True)

# ---- tune XGB on first fold (Optuna, subsample for speed)
TESTS=[2023,2024,2025,2026]
tr=g[g.year<TESTS[0]-1]; va=g[g.year==TESTS[0]-1]
trs=tr if DEVICE=="cuda" else tr.sample(min(len(tr),60000),random_state=0)
def objective(t):
    p=dict(max_depth=t.suggest_int("max_depth",3,8),learning_rate=0.05,min_child_weight=t.suggest_int("min_child_weight",5,200,log=True),
           subsample=t.suggest_float("subsample",0.5,1),colsample_bytree=t.suggest_float("colsample_bytree",0.2,0.9),
           reg_lambda=t.suggest_float("reg_lambda",0.1,50,log=True),gamma=t.suggest_float("gamma",0,5))
    m=fit_xgb(trs,va,S5,p); return ll(m.predict_proba(va[S5]),va.result.values)
st=optuna.create_study(direction="minimize"); st.optimize(objective,n_trials=int(sys.argv[3]) if len(sys.argv)>3 else 20)
BEST=st.best_params; log("optuna best",BEST,round(st.best_value,4))

rows=[]; allpred=[]
for T in TESTS:
    tr=g[g.year<T-1]; va=g[g.year==T-1]; te=g[g.year==T]
    if te.empty: continue
    P={}; V={}
    for name,fs in SETS.items():
        m=fit_xgb(tr,va,fs); P[name]=m.predict_proba(te[fs]); V[name]=m.predict_proba(va[fs]); log(T,name,m.best_iteration)
    # tuned + recency weights
    w=0.5**((T-1-tr.year)/3.0)
    m=fit_xgb(tr,va,S5,BEST,w=w.values); P["7. XGBoost ajustado (Optuna) + peso p/ jogos recentes"]=m.predict_proba(te[S5]); V["xgb_t"]=m.predict_proba(va[S5])
    # LightGBM
    lm=lgb.LGBMClassifier(n_estimators=3000,learning_rate=0.03,num_leaves=31,min_child_samples=50,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=5,verbose=-1,n_jobs=NJ)
    lm.fit(tr[S5],tr.result,sample_weight=w.values,eval_set=[(va[S5],va.result)],callbacks=[lgb.early_stopping(150,verbose=False)])
    P["8. LightGBM"]=lm.predict_proba(te[S5]); V["lgb"]=lm.predict_proba(va[S5])
    # Poisson goals model (Dixon-Coles)
    mh=fit_xgb(tr,va,S5,BEST,w=w.values,obj="count:poisson",target="hg"); ma=fit_xgb(tr,va,S5,BEST,w=w.values,obj="count:poisson",target="ag")
    lhv,lav=mh.predict(va[S5]),ma.predict(va[S5]); lht,lat=mh.predict(te[S5]),ma.predict(te[S5])
    rho=minimize_scalar(lambda r: ll(dc_probs(lhv,lav,r),va.result.values),bounds=(-0.3,0.3),method="bounded").x
    P["9. Modelo de gols (Poisson/Dixon-Coles)"]=dc_probs(lht,lat,rho); V["pois"]=dc_probs(lhv,lav,rho)
    # Ensemble + temperature calibration (weights and T chosen on validation year)
    best=(9,None)
    for a in np.arange(0,1.01,0.1):
        for b in np.arange(0,1.01-a,0.1):
            pv=a*V["xgb_t"]+b*V["lgb"]+(1-a-b)*V["pois"]; s=ll(pv,va.result.values)
            if s<best[0]: best=(s,(a,b))
    a,b=best[1]; ens_v=a*V["xgb_t"]+b*V["lgb"]+(1-a-b)*V["pois"]
    Tt=minimize_scalar(lambda t: ll(temp(ens_v,t),va.result.values),bounds=(0.5,2),method="bounded").x
    ens=temp(a*P["7. XGBoost ajustado (Optuna) + peso p/ jogos recentes"]+b*P["8. LightGBM"]+(1-a-b)*P["9. Modelo de gols (Poisson/Dixon-Coles)"],Tt)
    P["10. Ensemble (7+8+9) calibrado"]=ens; log(T,"ensemble weights",round(a,1),round(b,1),"T",round(Tt,3),"rho",round(rho,3))
    # Odds as features (only rows with odds)
    tro,vao=tr[tr.oH.notna()],va[va.oH.notna()]
    mo=fit_xgb(tro,vao,S5+ODDS,BEST,w=(0.5**((T-1-tro.year)/3.0)).values); P["11. XGBoost + odds como variável"]=mo.predict_proba(te[S5+ODDS])
    # stacking: odds + ensemble, logistic on validation
    vv=va.oH.notna().values; Zv=np.log(np.column_stack([va[ODDS].values[vv],ens_v[vv]]).clip(1e-6))
    st_lr=LogisticRegression(C=1.0,max_iter=2000).fit(Zv,va.result.values[vv])
    tt=te.oH.notna().values; Zt=np.log(np.column_stack([te[ODDS].fillna(1/3).values,ens]).clip(1e-6))
    P["12. Odds + ensemble (stacking)"]=st_lr.predict_proba(Zt)
    P["Casa de aposta (Pinnacle, fechamento)"]=te[ODDS].values
    yt=te.result.values; mask=te.oH.notna().values
    for k,p in P.items():
        r={"ano":T,"modelo":k,"jogos_com_odds":int(mask.sum()),"acerto":(p[mask].argmax(1)==yt[mask]).mean(),"logloss":ll(p[mask],yt[mask]),"rps":rps(p[mask],yt[mask])}
        if "Casa" not in k and "odds" not in k.lower(): r["logloss_todos"]=ll(p,yt); r["acerto_todos"]=(p.argmax(1)==yt).mean(); r["jogos_todos"]=len(yt)
        rows.append(r)
    ap=te[["fixture_id","date","league_id","home_team_id","away_team_id","hg","ag","result","oH","oD","oA","home_win","draw","away_win"]].copy()
    for k in ("10. Ensemble (7+8+9) calibrado","12. Odds + ensemble (stacking)","11. XGBoost + odds como variável"):
        ap[[f"m{k.split('.')[0]}_{c}" for c in "HDA"]]=P[k]
    allpred.append(ap)
    pd.DataFrame(rows).to_csv(OUTD+"/results2.csv",index=False); log("fold done",T)
    if T==2025:
        imp=pd.Series(m.get_booster().get_score(importance_type="gain")).sort_values(ascending=False); imp.to_csv(OUTD+"/importance2.csv")
R=pd.DataFrame(rows); pd.concat(allpred).to_parquet(OUTD+"/preds2.parquet",index=False)
W=R.groupby("modelo").apply(lambda d: pd.Series({"jogos":d.jogos_com_odds.sum(),"acerto":np.average(d.acerto,weights=d.jogos_com_odds),"logloss":np.average(d.logloss,weights=d.jogos_com_odds),"rps":np.average(d.rps,weights=d.jogos_com_odds)}))
pd.set_option("display.width",220); print(W.sort_values("logloss").round(4)); W.to_csv(OUTD+"/summary2.csv")

pr=pd.concat(allpred); pr=pr[pr.oH.notna()]; yv=pr.result.values; od=pr[["home_win","draw","away_win"]].values
bet_rows=[]
for k in ("m10","m12"):
    mp=pr[[f"{k}_{c}" for c in "HDA"]].values
    for edge in (0.0,0.02,0.05,0.10):
        bet=(mp*od-1)>edge; profit=np.where(np.eye(3)[yv].astype(bool)&bet,od,0)-bet
        bet_rows.append({"modelo":k,"edge":edge,"apostas":int(bet.sum()),"lucro":float(profit.sum()),"roi":float(profit.sum()/max(bet.sum(),1))})
B=pd.DataFrame(bet_rows); B.to_csv(OUTD+"/betting.csv",index=False); print(B.round(4))
