"""reward-lab command line.

  reward-lab loop     --exp configs/experiment.yaml --backend kaggle
  reward-lab finals   --exp configs/experiment.yaml --backend kaggle
  reward-lab analyze  --exp configs/experiment.yaml
  reward-lab smoke    (whole loop on CPU with configs/smoke.yaml)
"""

import argparse
import json
import subprocess
import time
from importlib.metadata import version

from . import remote, train
from .loop import Experiment
from .tasks import ROOT


def _train(exp, jobs, backend, slug):
    print(f"training {len(jobs)} runs on {backend}", flush=True)
    if backend == "local":
        train.run_jobs(jobs, exp.dir / "train", impl="jax", smoke=exp.smoke)
    else:
        remote.run(jobs, exp.dir / "train", exp.dir / "kaggle_logs", slug=slug)


def _stamp(exp, cmd):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip() or "none"
    info = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": cmd, "commit": commit, "config": exp.cfg,
            "versions": {p: version(p) for p in ("jax", "mujoco", "playground", "brax")}}
    exp.dir.mkdir(parents=True, exist_ok=True)
    with open(exp.dir / "invocations.jsonl", "a") as f:
        f.write(json.dumps(info) + "\n")


def loop(exp, backend, slug):
    _stamp(exp, "loop")
    while True:
        n = exp.generate_ready()
        jobs = exp.pending_jobs()
        print(f"generated {n} samples, {len(jobs)} runs to train", flush=True)
        if jobs:
            _train(exp, jobs, backend, slug)
            # A run that is still missing after a remote batch means the
            # kernel died; stop rather than loop forever on it.
            if backend != "local" and len(exp.pending_jobs()) >= len(jobs):
                raise SystemExit("no progress from the remote batch, check kaggle_logs/")
        elif n == 0:
            break
    print("loop finished", flush=True)


def finals(exp, backend, slug):
    _stamp(exp, "finals")
    jobs = exp.final_jobs()
    if jobs:
        _train(exp, jobs, backend, slug)


def main():
    p = argparse.ArgumentParser(prog="reward-lab")
    p.add_argument("cmd", choices=["loop", "finals", "analyze", "smoke"])
    p.add_argument("--exp", default=str(ROOT / "configs" / "experiment.yaml"))
    p.add_argument("--backend", choices=["local", "kaggle"], default="local")
    p.add_argument("--slug", default="reward-lab-worker", help="Kaggle kernel name (one per parallel batch)")
    a = p.parse_args()
    if a.cmd == "smoke":
        exp = Experiment(ROOT / "configs" / "smoke.yaml")
        loop(exp, "local", a.slug)
        finals(exp, "local", a.slug)
        return
    exp = Experiment(a.exp)
    if a.cmd == "loop":
        loop(exp, a.backend, a.slug)
    elif a.cmd == "finals":
        finals(exp, a.backend, a.slug)
    else:
        from .analysis import run as analyze
        analyze(exp)


if __name__ == "__main__":
    main()
