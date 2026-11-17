"""Helpers shared by the FutData updaters: HTTP with retries and merging tables into the HF dataset."""
import io, json, os, time, urllib.error, urllib.request
import pandas as pd

REPO = os.environ.get("HF_REPO", "Alisson990/futdata")
OUT = os.environ.get("FUTDATA_OUT", "out")


def http_json(url, headers=None, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "futdata-updater", **(headers or {})})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r), dict(r.headers)
        except urllib.error.HTTPError as e:
            wait = 300 if e.code == 429 else 15
            print(f"HTTP {e.code} on {url.split('?')[0]}: {e.read()[:200]!r}; retry in {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:  # network hiccup
            print(f"error {e}; retry", flush=True)
            time.sleep(15)
    raise RuntimeError(f"failed after {tries} tries: {url.split('?')[0]}")


def http_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _clean(df):
    for c in df.columns:
        if df[c].dtype == object:
            if df[c].map(lambda v: isinstance(v, (list, dict))).any():
                df[c] = df[c].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
            df[c] = df[c].astype("string")
    return df


def existing(rel):
    """Current version of data/<rel>.parquet from the Hub (empty frame if missing)."""
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(REPO, f"data/{rel}.parquet", repo_type="dataset")
        return pd.read_parquet(p)
    except Exception as e:
        print(f"no existing {rel}: {type(e).__name__}", flush=True)
        return pd.DataFrame()


def merge_save(rel, new, keys, merge=True):
    """Upsert `new` into data/<rel>.parquet (newest row wins on `keys`) and write it under OUT."""
    if new is None or new.empty:
        print(f"{rel}: nothing new", flush=True)
        return
    new = _clean(new.copy())
    df = new
    if merge:
        old = existing(rel)
        if not old.empty:
            old = _clean(old)
            df = pd.concat([old, new], ignore_index=True)
            for c in df.columns:  # unify dtypes after concat
                if df[c].dtype == object:
                    df[c] = df[c].astype("string")
    if merge:
        df = df.drop_duplicates(subset=[k for k in keys if k in df.columns], keep="last")
    else:  # full replacement: only drop rows that are identical
        df = df.drop_duplicates()
    path = f"{OUT}/data/{rel}.parquet"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_parquet(path, index=False, compression="zstd")
    except Exception:  # mixed types: fall back to strings
        df.astype("string").to_parquet(path, index=False, compression="zstd")
    print(f"{rel}: +{len(new)} -> {len(df)} rows", flush=True)


def upload(message):
    from huggingface_hub import HfApi
    if not os.path.isdir(f"{OUT}/data"):
        print("nothing to upload")
        return
    HfApi().upload_folder(repo_id=REPO, repo_type="dataset", folder_path=OUT, commit_message=message)
    print("uploaded", flush=True)
