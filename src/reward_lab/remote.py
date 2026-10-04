"""Run a batch of training jobs on a free Kaggle GPU (2x T4) and pull the results.

The LLM side of the loop stays on the local machine; only PPO runs remotely.
Code and jobs travel inside the kernel script itself (base64 tarball), so
there is no dataset to version and nothing to push to GitHub. Each GPU gets
its own worker process (CUDA_VISIBLE_DEVICES), which roughly doubles
throughput compared to one pmap'ed run, since a run on its own is latency
bound rather than compute bound at these sizes.

Requires a Kaggle token (~/.kaggle) and a phone-verified account (internet
access is needed in the kernel for pip).
"""

import base64
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from .tasks import ROOT

KERNEL = '''
import base64, io, json, os, subprocess, sys, tarfile, time
BUNDLE = "{bundle}"
t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "playground==0.2.0", "jax[cuda12]==0.8.0"], check=True)
lab = "/tmp/lab"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE))).extractall(lab)
os.makedirs("/kaggle/working/logs", exist_ok=True)
gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()
print("setup", round(time.time() - t0), "s", gpus, flush=True)
procs = []
for i in range(max(1, len(gpus))):
    env = dict(os.environ, PYTHONPATH=lab + "/src", CUDA_VISIBLE_DEVICES=str(i))
    log = open(f"/kaggle/working/logs/worker{{i}}.log", "w")
    procs.append(subprocess.Popen([sys.executable, "-m", "reward_lab.train", "--jobs", lab + "/jobs.json",
                                   "--out", "/kaggle/working/out", "--impl", "{impl}",
                                   "--shard", f"{{i}}/{{max(1, len(gpus))}}"], env=env, stdout=log, stderr=subprocess.STDOUT))
for p in procs:
    p.wait()
print("done", round(time.time() - t0), "s", flush=True)
'''


def _bundle(jobs):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(ROOT / "src" / "reward_lab", "src/reward_lab",
                filter=lambda ti: None if "__pycache__" in ti.name else ti)
        tar.add(ROOT / "configs", "configs")
        data = json.dumps(jobs).encode()
        ti = tarfile.TarInfo("jobs.json")
        ti.size = len(data)
        tar.addfile(ti, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def _kaggle(*args, retries=5):
    for i in range(retries):
        r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
        if r.returncode == 0 and "Connection aborted" not in r.stdout + r.stderr:
            return r.stdout
        print("kaggle", args[:2], "failed:", (r.stdout + r.stderr).strip()[-200:], flush=True)
        time.sleep(20 * (i + 1))
    raise RuntimeError(f"kaggle {args} kept failing")


def username():
    out = _kaggle("config", "view")
    return next(l.split(":", 1)[1].strip() for l in out.splitlines() if "username" in l)


def submit(jobs, slug, impl="warp"):
    # Interleave tasks so that the GPU shards get similar loads.
    jobs = sorted(jobs, key=lambda j: (j["task"], j["key"]))
    d = Path(tempfile.mkdtemp())
    (d / "run.py").write_text(KERNEL.format(bundle=_bundle(jobs), impl=impl))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username()}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": []}))
    print(_kaggle("kernels", "push", "-p", str(d), "--accelerator", "NvidiaTeslaT4").strip().splitlines()[-1])
    shutil.rmtree(d)


def wait(slug, poll=60):
    ref = f"{username()}/{slug}"
    while True:
        out = _kaggle("kernels", "status", ref)
        if "COMPLETE" in out or "ERROR" in out or "CANCEL" in out:
            return out.strip()
        time.sleep(poll)


def fetch(slug, train_dir: Path, log_dir: Path):
    """Copy every finished run into train_dir. Returns the number of runs."""
    tmp = Path(tempfile.mkdtemp())
    _kaggle("kernels", "output", f"{username()}/{slug}", "-p", str(tmp))
    n = 0
    for m in (tmp / "out").glob("*/*/metrics.json"):
        dst = train_dir / m.parent.parent.name / m.parent.name
        if not (dst / "metrics.json").exists():
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(m.parent, dst)
            n += 1
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for f in list((tmp / "logs").glob("*.log")) + list(tmp.glob("*.log")):
        shutil.copy(f, log_dir / f"{stamp}_{slug}_{f.name}")
    shutil.rmtree(tmp)
    return n


def run(jobs, train_dir: Path, log_dir: Path, slug="reward-lab-worker", impl="warp"):
    submit(jobs, slug, impl)
    time.sleep(30)
    print("kaggle:", wait(slug), flush=True)
    n = fetch(slug, train_dir, log_dir)
    print(f"kaggle: fetched {n}/{len(jobs)} runs", flush=True)
    return n
