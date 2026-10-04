"""The Eureka-style loop. Stateless: everything is read back from results/.

Layout of results/<experiment>/:
  llm_cache/<sha>.json                     every LLM answer (prompt hash -> answer)
  runs/<task>/s<seed>/<cond>/it<i>/c<k>/   one LLM sample: answer.md, reward.py, gen.json
  runs/<task>/s<seed>/<cond>/it<i>/prompt.md, feedback/*.png   what the LLM was shown
  train/<task>/<key>/                      one PPO run: metrics.json, traj.npz
  finals.json                              which rewards go to the full-budget finals

A training run is identified by hash(task, code, seed, steps), so identical
rewards are trained once. This matters because all conditions share
iteration 0: same prompt and same sampling seeds, hence the same K first
candidates (cached LLM answers, cached training). B and C differ from
iteration 1 on, and only by the images.

Condition A gets K*N samples of the iteration-0 prompt, which is the same
number of trained candidates as B and C.
"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from . import feedback, sandbox
from .context import describe
from .llm import LLM
from .tasks import ROOT, load_task

PROMPTS = ROOT / "prompts"
VERDICT_RE = re.compile(r"Candidate\s*(\d+)\b[^\n]*?VERDICT:\W*(solves|hacks|fails)", re.I)


def _read(name):
    return (PROMPTS / name).read_text()


class Experiment:
    def __init__(self, cfg_path, results=None):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text())
        self.name = Path(cfg_path).stem
        self.dir = Path(results or ROOT / "results") / self.name
        self.K, self.N = self.cfg["K"], self.cfg["N"]
        self.smoke = bool(self.cfg.get("smoke", False))
        llm_cfg = self.cfg["llm"]
        self.llm = LLM(llm_cfg["model"], llm_cfg["temperature"], self.dir / "llm_cache")
        self.tasks = {n: load_task(n) for n in self.cfg["tasks"]}
        self._ctx = {}

    # ---------- budgets and identifiers ----------

    def budget(self, task, final=False):
        if self.smoke:
            return 20_000, 2
        return (task.full_steps, 11) if final else (task.short_steps, 6)

    def job(self, task, code, comps, seed, final=False):
        steps, num_evals = self.budget(task, final)
        key = hashlib.sha1(json.dumps([task.name, code, seed, steps]).encode()).hexdigest()[:12]
        return {"task": task.name, "code": code, "comps": comps, "seed": seed,
                "steps": steps, "num_evals": num_evals, "key": key}

    def metrics(self, task_name, key):
        p = self.dir / "train" / task_name / key / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else None

    def runs(self):
        for t in self.tasks.values():
            for s in self.cfg["seeds"]:
                for c in self.cfg["conditions"]:
                    yield t, s, c

    def n_iters(self, cond):
        return 1 if cond == "A" else self.N

    def n_samples(self, cond, it):
        return self.K * self.N if cond == "A" else self.K

    def n_shared(self):
        return max(self.n_samples(c, 0) for c in self.cfg["conditions"])

    def it_dir(self, task, seed, cond, it):
        # Iteration 0 is generated once and shared by all conditions: B and C
        # use its first K samples, A uses all K*N.
        if it == 0:
            return self.dir / "runs" / task.name / f"s{seed}" / "it0"
        return self.dir / "runs" / task.name / f"s{seed}" / cond / f"it{it}"

    def ctx_batch(self, task):
        if task.name not in self._ctx:
            self._ctx[task.name] = sandbox.make_ctx_batch(task)
        return self._ctx[task.name]

    # ---------- reading back candidates ----------

    def candidates(self, task, seed, cond, it):
        """List of dicts (one per sample) for a generated iteration, with metrics if trained."""
        out = []
        for k in range(self.n_samples(cond, it)):
            d = self.it_dir(task, seed, cond, it) / f"c{k}"
            g = json.loads((d / "gen.json").read_text())
            g.update(dir=d, it=it, k=k)
            if g["status"] == "ok":
                g["code"] = (d / "reward.py").read_text()
                g["metrics"] = self.metrics(task.name, g["key"])
            out.append(g)
        return out

    def generated(self, task, seed, cond, it):
        d = self.it_dir(task, seed, cond, it)
        return all((d / f"c{k}" / "gen.json").exists() for k in range(self.n_samples(cond, it)))

    def pending(self, task, seed, cond, it):
        return [c for c in self.candidates(task, seed, cond, it)
                if c["status"] == "ok" and c["metrics"] is None]

    @staticmethod
    def fitness(c):
        m = c.get("metrics")
        return m["fitness"] if m and m.get("status") == "ok" else -np.inf

    # ---------- prompts ----------

    def initial_messages(self, task):
        user = _read("task.md").format(description=task.description, context=describe(task.family))
        return [{"role": "system", "content": _read("system.md")}, {"role": "user", "content": user}]

    def feedback_messages(self, task, seed, cond, it):
        """Feedback on iteration it-1 (all K candidates), plus the best candidate
        so far if it comes from an earlier iteration."""
        last = self.candidates(task, seed, cond, it - 1)
        history = [c for i in range(it - 1) for c in self.candidates(task, seed, cond, i)]
        shown = list(last)
        best = max(last + history, key=self.fitness)
        if not any(best is c for c in last) and np.isfinite(self.fitness(best)):
            shown.append(best)

        d = self.it_dir(task, seed, cond, it)
        (d / "feedback").mkdir(parents=True, exist_ok=True)
        steps = self.budget(task)[0]
        blocks, images, shown_meta = [], [], []
        for n, c in enumerate(shown, 1):
            title = f"### Candidate {n}"
            if c is best:
                title += " (best task score so far)"
            if c["it"] != it - 1:
                title += " (from an earlier iteration)"
            lines = [title]
            if c["status"] != "ok":
                lines.append(f"This reward was rejected before training: {c['errors'][-1][:400]}")
            else:
                lines += ["```python", c["code"].strip(), "```"]
                m = c["metrics"]
                lines.append(feedback.summarize(m, c["comps"]))
                if cond == "C" and m.get("status") == "ok":
                    tr = dict(np.load(self.dir / "train" / task.name / c["key"] / "traj.npz"))
                    png = d / "feedback" / f"candidate{n}.png"
                    if not png.exists():
                        feedback.contact_sheet(task, tr, f"Candidate {n}").save(png)
                    images.append(str(png))
                    lines.append(f"Image: attached image #{len(images)}.")
            blocks.append("\n".join(lines))
            shown_meta.append({"n": n, "it": c["it"], "k": c["k"], "key": c.get("key")})

        if cond == "C":
            evidence = ("For each candidate you get its reward code, its training statistics, and one "
                        "attached image: 8 frames of the trained policy spread over the 10 s episode, "
                        "with the time and the running task score under each frame.")
        else:
            evidence = "For each candidate you get its reward code and its training statistics."
        user = _read("feedback.md").format(
            description=task.description, context=describe(task.family), steps=f"{steps:,}",
            score_doc=task.score_doc, evidence=evidence, candidates="\n\n".join(blocks))
        msgs = [{"role": "system", "content": _read("system.md")},
                {"role": "user", "content": user, "images": images}]
        (d / "shown.json").write_text(json.dumps(shown_meta, indent=1))
        return msgs

    # ---------- generation ----------

    def sample(self, task, seed, cond, it, k, messages):
        d = self.it_dir(task, seed, cond, it) / f"c{k}"
        if (d / "gen.json").exists():
            return
        d.mkdir(parents=True, exist_ok=True)
        # Same seed for the same (loop seed, iteration, index) in every
        # condition: this is what makes iteration 0 identical across A/B/C.
        sample_seed = seed * 10000 + it * 100 + k
        msgs = list(messages)
        log = {"status": "error", "errors": [], "calls": []}
        answers = []
        for attempt in range(self.cfg["max_fixes"] + 1):
            out = self.llm.chat(msgs, sample_seed)
            answers.append(out["content"])
            log["calls"].append({k_: out[k_] for k_ in ("model", "prompt_tokens", "completion_tokens",
                                                        "latency_s", "cached", "n_images")})
            try:
                code = sandbox.extract_code(out["content"])
                comps = sandbox.validate(code, self.ctx_batch(task))
            except sandbox.RewardError as e:
                log["errors"].append(str(e))
                msgs = msgs + [{"role": "assistant", "content": out["content"]},
                               {"role": "user", "content": _read("fix.md").format(error=str(e))}]
                continue
            job = self.job(task, code, comps, seed)
            log.update(status="ok", comps=comps, key=job["key"], fixes=attempt)
            (d / "reward.py").write_text(code)
            break
        log["verdicts"] = {int(n): v.lower() for n, v in VERDICT_RE.findall(answers[0])} if it > 0 else {}
        (d / "answer.md").write_text("\n\n---- FIX ----\n\n".join(answers))
        (d / "gen.json").write_text(json.dumps(log, indent=1))

    def generate_ready(self, workers=4):
        """Generate every iteration whose inputs are available. Returns how many."""
        todo, seen = [], set()
        for task, seed, cond in self.runs():
            for it in range(self.n_iters(cond)):
                if self.generated(task, seed, cond, it):
                    if self.pending(task, seed, cond, it):
                        break
                    continue
                msgs = self.initial_messages(task) if it == 0 else self.feedback_messages(task, seed, cond, it)
                d = self.it_dir(task, seed, cond, it)
                if d not in seen:
                    seen.add(d)
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "prompt.md").write_text(msgs[-1]["content"])
                    n = self.n_shared() if it == 0 else self.n_samples(cond, it)
                    todo += [(task, seed, cond, it, k, msgs) for k in range(n)]
                break
        # Validation compiles on CPU; build the sample states before threading.
        for t in {x[0].name for x in todo}:
            self.ctx_batch(self.tasks[t])
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda a: self.sample(*a), todo))
        return len(todo)

    def pending_jobs(self):
        jobs = {}
        for task in self.tasks.values():
            if task.human_reward:
                comps = sandbox.validate(task.human_reward, self.ctx_batch(task))
                for s in self.cfg["seeds"]:
                    j = self.job(task, task.human_reward, comps, s)
                    if self.metrics(task.name, j["key"]) is None:
                        jobs[j["key"]] = j
        for task, seed, cond in self.runs():
            for it in range(self.n_iters(cond)):
                if not self.generated(task, seed, cond, it):
                    break
                for c in self.pending(task, seed, cond, it):
                    jobs[c["key"]] = self.job(task, c["code"], c["comps"], seed)
        return list(jobs.values())

    def done(self):
        return all(self.generated(t, s, c, self.n_iters(c) - 1) and not self.pending(t, s, c, self.n_iters(c) - 1)
                   for t, s, c in self.runs())

    # ---------- finals ----------

    def all_candidates(self, task, cond):
        return [c for s in self.cfg["seeds"] for it in range(self.n_iters(cond))
                for c in self.candidates(task, s, cond, it)]

    def final_jobs(self):
        """Best reward of each condition (over loop seeds), plus the human one,
        retrained with the full budget on fresh seeds."""
        jobs, table = [], {}
        for task in self.tasks.values():
            entries = {}
            for cond in self.cfg["conditions"]:
                best = max(self.all_candidates(task, cond), key=self.fitness)
                entries[cond] = {"code": best["code"], "comps": best["comps"],
                                 "source": str(best["dir"].relative_to(self.dir)),
                                 "short_fitness": self.fitness(best)}
            if task.human_reward:
                comps = sandbox.validate(task.human_reward, self.ctx_batch(task))
                entries["human"] = {"code": task.human_reward, "comps": comps, "source": "human"}
            table[task.name] = {}
            for name, e in entries.items():
                keys = []
                for s in self.cfg["final_seeds"]:
                    j = self.job(task, e["code"], e["comps"], 100 + s, final=True)
                    keys.append(j["key"])
                    jobs.append(j)
                table[task.name][name] = {**{k: v for k, v in e.items() if k != "code"}, "keys": keys}
        (self.dir / "finals.json").write_text(json.dumps(table, indent=1))
        uniq = {j["key"]: j for j in jobs if self.metrics(j["task"], j["key"]) is None}
        return list(uniq.values())
