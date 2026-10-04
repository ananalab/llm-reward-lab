"""Tables and figures from results/<experiment>/. Every number in the report comes from here.

Definitions (also in the report):
* normalised fitness: fitness / fitness of the human reward trained with the
  same budget and seed (cartpole, cheetah_run). cheetah_stand has no human
  reward and is reported raw.
* label of a trained candidate (automatic):
    solves  fitness >= task success threshold;
    hacks   not solves, the reward clearly went up during training
            ((R_last - R_first) / max|R| > 0.2) while the fitness did not
            (F_last - F_first < 0.25 * threshold);
    fails   everything else (learned little, or partial progress).
  "hacking rate" = share of trained candidates labelled hacks.
* condition A at "iteration i": expected best fitness of K*(i+1) samples drawn
  without replacement from A's K*N pool (order-free, so no luck of ordering).
"""

import json
from collections import defaultdict
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .loop import Experiment  # noqa: E402
from .tasks import ROOT  # noqa: E402

COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "human": "#52514e"}
NAMES = {"A": "A: no feedback", "B": "B: text feedback", "C": "C: text + images", "human": "human reward"}


def label(m, task):
    if not m or m.get("status") != "ok":
        return "crash"
    if m["fitness"] >= task.success:
        return "solves"
    c = m["curve"]
    r = [x["reward"] for x in c]
    f = [x["fitness"] for x in c]
    r_gain = (r[-1] - r[0]) / (max(abs(x) for x in r) + 1e-8)
    f_gain = f[-1] - f[0]
    return "hacks" if r_gain > 0.2 and f_gain < 0.25 * task.success else "fails"


def expected_best(values, n):
    """E[max] of n draws without replacement from values."""
    v = np.sort(np.asarray(values, float))
    M = len(v)
    n = min(n, M)
    w = np.array([comb(j, n - 1) for j in range(M)], float) / comb(M, n)
    return float((w * v).sum())


class Report:
    def __init__(self, exp: Experiment):
        self.exp = exp
        self.out = {}
        self.fig_dir = ROOT / "report" / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    def human(self, task, seed, final=False):
        if not task.human_reward:
            return None
        from . import sandbox
        comps = sandbox.validate(task.human_reward, self.exp.ctx_batch(task))
        m = self.exp.metrics(task.name, self.exp.job(task, task.human_reward, comps, seed, final)["key"])
        return m["fitness"] if m and m.get("status") == "ok" else None

    def norm(self, task, seed, f):
        h = self.human(task, seed)
        return f / h if h else f

    # ----- best fitness per iteration -----
    def iterations(self):
        e = self.exp
        res = {}
        for task in e.tasks.values():
            res[task.name] = {}
            for cond in e.cfg["conditions"]:
                per_seed = []
                for s in e.cfg["seeds"]:
                    # A rejected or crashed candidate gives no policy: fitness 0.
                    fit = lambda c: e.fitness(c) if np.isfinite(e.fitness(c)) else 0.0  # noqa: E731
                    if cond == "A":
                        pool = [fit(c) for c in e.candidates(task, s, "A", 0)]
                        curve = [expected_best(pool, e.K * (i + 1)) for i in range(e.N)]
                    else:
                        best, curve = -np.inf, []
                        for it in range(e.N):
                            best = max([best] + [fit(c) for c in e.candidates(task, s, cond, it)])
                            curve.append(best)
                    per_seed.append([self.norm(task, s, x) for x in curve])
                res[task.name][cond] = per_seed
        self.out["iterations"] = res
        return res

    # ----- validity, labels, detection, tokens -----
    def generation_stats(self):
        e = self.exp
        stats = defaultdict(lambda: defaultdict(float))
        labels = {}
        for task, s, cond in e.runs():
            for it in range(e.n_iters(cond)):
                for c in e.candidates(task, s, cond, it):
                    st = stats[cond]
                    st["samples"] += 1
                    st["valid_first_try"] += c["status"] == "ok" and c.get("fixes") == 0
                    st["valid_after_fixes"] += c["status"] == "ok"
                    st["llm_calls"] += len(c["calls"])
                    st["prompt_tokens"] += sum(x["prompt_tokens"] for x in c["calls"])
                    st["completion_tokens"] += sum(x["completion_tokens"] for x in c["calls"])
                    st["images"] += sum(x["n_images"] for x in c["calls"])
                    if c["status"] == "ok" and c["metrics"]:
                        lab = label(c["metrics"], task)
                        labels[(task.name, c["key"])] = lab
                        if not (cond != "A" and it == 0):  # iteration 0 is shared: count it once, under A
                            st[f"label_{lab}"] += 1
                            st["trained"] += 1
        self.out["generation"] = {k: dict(v) for k, v in stats.items()}
        self.labels = labels
        return self.out["generation"]

    def detection(self, manual=None):
        """LLM verdicts (B and C, iterations >= 1) vs labels."""
        e = self.exp
        rows = []
        for task, s, cond in e.runs():
            if cond == "A":
                continue
            for it in range(1, e.N):
                d = e.it_dir(task, s, cond, it)
                shown = {x["n"]: x for x in json.loads((d / "shown.json").read_text())}
                for c in e.candidates(task, s, cond, it):
                    for n, verdict in c.get("verdicts", {}).items():
                        key = shown.get(int(n), {}).get("key")
                        truth = self.labels.get((task.name, key))
                        if truth in ("solves", "hacks", "fails"):
                            rows.append({"task": task.name, "cond": cond, "seed": s, "it": it, "key": key,
                                         "verdict": verdict, "auto": truth,
                                         "manual": (manual or {}).get(f"{task.name}/{key}")})
        out = {}
        for cond in ("B", "C"):
            r = [x for x in rows if x["cond"] == cond]
            if not r:
                continue
            acc = np.mean([x["verdict"] == x["auto"] for x in r])
            hk = [x for x in r if x["auto"] == "hacks"]
            pred_h = [x for x in r if x["verdict"] == "hacks"]
            out[cond] = {"n": len(r), "accuracy": float(acc),
                         "hack_recall": float(np.mean([x["verdict"] == "hacks" for x in hk])) if hk else None,
                         "n_hacks": len(hk),
                         "hack_precision": float(np.mean([x["auto"] == "hacks" for x in pred_h])) if pred_h else None,
                         "n_pred_hacks": len(pred_h)}
            m = [x for x in r if x["manual"]]
            if m:
                out[cond]["manual_n"] = len(m)
                out[cond]["manual_accuracy"] = float(np.mean([x["verdict"] == x["manual"] for x in m]))
        self.out["detection"] = out
        self.detection_rows = rows
        return out

    # ----- finals -----
    def finals(self):
        e = self.exp
        p = e.dir / "finals.json"
        if not p.exists():
            return None
        table = json.loads(p.read_text())
        res = {}
        for tname, entries in table.items():
            task = e.tasks[tname]
            res[tname] = {}
            for name, ent in entries.items():
                fits = [m["fitness"] for k in ent["keys"]
                        if (m := e.metrics(tname, k)) and m.get("status") == "ok"]
                res[tname][name] = {"fitness": fits, "source": ent["source"],
                                    "short_fitness": ent.get("short_fitness")}
            if "human" in res[tname] and res[tname]["human"]["fitness"]:
                h = np.mean(res[tname]["human"]["fitness"])
                for v in res[tname].values():
                    v["normalised"] = [f / h for f in v["fitness"]]
        self.out["finals"] = res
        return res

    def budget_check(self):
        """Does the short-budget checkpoint rank the final runs like the full budget does?"""
        from scipy.stats import kendalltau, spearmanr
        e = self.exp
        if not (e.dir / "finals.json").exists():
            return None
        table = json.loads((e.dir / "finals.json").read_text())
        res = {}
        for tname, entries in table.items():
            task = e.tasks[tname]
            short, full = [], []
            for ent in entries.values():
                for k in ent["keys"]:
                    m = e.metrics(tname, k)
                    if not m or m.get("status") != "ok":
                        continue
                    c = m["curve"]
                    i = int(np.argmin([abs(x["step"] - task.short_steps) for x in c]))
                    short.append(c[i]["fitness"])
                    full.append(c[-1]["fitness"])
            if len(short) > 3:
                res[tname] = {"n_runs": len(short), "kendall_tau": float(kendalltau(short, full)[0]),
                              "spearman": float(spearmanr(short, full)[0])}
        self.out["budget_check"] = res
        return res

    # ----- figures -----
    def fig_iterations(self):
        res = self.out["iterations"]
        tasks = list(res)
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.4), squeeze=False)
        for ax, t in zip(axes[0], tasks):
            for cond, per_seed in res[t].items():
                arr = np.array(per_seed, float)
                x = np.arange(1, arr.shape[1] + 1)
                for row in arr:
                    ax.plot(x, row, color=COLORS[cond], lw=1, alpha=0.3)
                ax.plot(x, arr.mean(0), color=COLORS[cond], lw=2, marker="o", ms=5, label=NAMES[cond])
            if self.exp.tasks[t].human_reward:
                ax.axhline(1.0, color=COLORS["human"], lw=1, ls="--", label="human reward (same budget)")
                ax.set_ylabel("best fitness / human")
            else:
                ax.set_ylabel("best fitness (fraction of time reared up)")
            ax.set_title(t)
            ax.set_xlabel(f"iteration (x{self.exp.K} candidates)")
            ax.set_xticks(x)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0][0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(self.fig_dir / "best_fitness_per_iteration.png", dpi=150)
        plt.close(fig)

    def fig_finals(self):
        res = self.out.get("finals")
        if not res:
            return
        tasks = list(res)
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.2), squeeze=False)
        for ax, t in zip(axes[0], tasks):
            names = [n for n in ("human", "A", "B", "C") if n in res[t]]
            key = "normalised" if "human" in res[t] else "fitness"
            for i, n in enumerate(names):
                vals = res[t][n].get(key, [])
                if not vals:
                    continue
                ax.bar(i, np.mean(vals), width=0.6, color=COLORS[n], edgecolor="white", linewidth=2)
                ax.scatter(np.full(len(vals), i), vals, color="black", s=10, zorder=3)
            ax.set_xticks(range(len(names)), [n if n != "human" else "human" for n in names])
            ax.set_ylabel("fitness / human" if key == "normalised" else "fitness")
            ax.set_title(t)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(self.fig_dir / "finals.png", dpi=150)
        plt.close(fig)


def run(exp: Experiment):
    rep = Report(exp)
    manual_path = exp.dir / "manual_labels.json"
    manual = json.loads(manual_path.read_text()) if manual_path.exists() else None
    rep.iterations()
    rep.generation_stats()
    rep.detection({k: v["label"] for k, v in manual.items()} if manual else None)
    rep.finals()
    rep.budget_check()
    rep.fig_iterations()
    rep.fig_finals()
    out = exp.dir / "analysis.json"
    out.write_text(json.dumps(rep.out, indent=1, default=float))
    (exp.dir / "detection_rows.json").write_text(json.dumps(rep.detection_rows, indent=1))
    labels = [{"task": t, "key": k, "label": v} for (t, k), v in rep.labels.items()]
    (exp.dir / "labels.json").write_text(json.dumps(labels, indent=1))
    print(json.dumps({k: v for k, v in rep.out.items() if k != "iterations"}, indent=1, default=float))
    return rep
