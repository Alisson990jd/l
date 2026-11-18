"""Run futdata/ml/model3.py on a Modal GPU.

    modal run futdata/ml/modal_train.py --trials 60

Downloads the feature table from the Hugging Face dataset, trains every model on the GPU
(XGBoost device=cuda) and writes the result files to ./ml_results.
"""
import os
import modal

app = modal.App("futdata-train")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("xgboost==2.1.4", "lightgbm==4.5.0", "optuna==4.1.0", "scikit-learn==1.5.2",
                 "pandas==2.2.3", "pyarrow==18.1.0", "scipy==1.14.1", "huggingface_hub==0.26.5")
    .add_local_file(os.path.join(os.path.dirname(__file__), "model3.py"), "/root/model3.py")
)


@app.function(image=image, gpu="A10G", cpu=8.0, memory=32768, timeout=6 * 3600)
def train(trials: int = 60, repo: str = "Alisson990/futdata"):
    import glob, shutil, subprocess
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, "ml/features_m3.parquet", repo_type="dataset")
    c = hf_hub_download(repo, "ml/features_m3.parquet.cols.json", repo_type="dataset")
    shutil.copy(p, "/tmp/m3.parquet"); shutil.copy(c, "/tmp/m3.parquet.cols.json")
    env = {**os.environ, "XGB_DEVICE": "cuda", "N_JOBS": "8"}
    proc = subprocess.run(["python", "/root/model3.py", "/tmp/m3.parquet", "/tmp/out", str(trials)],
                          env=env, capture_output=True, text=True)
    log = proc.stdout + "\n--- stderr (last 5k) ---\n" + proc.stderr[-5000:]
    files = {os.path.basename(f): open(f, "rb").read() for f in glob.glob("/tmp/out/*")}
    files["train.log"] = log.encode()
    files["returncode.txt"] = str(proc.returncode).encode()
    return files


@app.local_entrypoint()
def main(trials: int = 60):
    files = train.remote(trials)
    os.makedirs("ml_results", exist_ok=True)
    for name, data in files.items():
        open(os.path.join("ml_results", name), "wb").write(data)
    print(files["train.log"].decode()[-3000:])
