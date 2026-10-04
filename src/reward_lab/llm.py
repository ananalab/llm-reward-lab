"""LLM client: Ollama chat API, with an on-disk cache.

Ollama serves both local models and its hosted "cloud" models through the
same endpoint, and its free tier includes gemma4:31b, which reads images.
That gives a multimodal model at zero cost, behind one HTTP call.

* Local daemon (signed in with `ollama signin`): OLLAMA_HOST=http://localhost:11434 (default)
* Directly against ollama.com (Colab/Kaggle): OLLAMA_HOST=https://ollama.com and
  OLLAMA_API_KEY in the environment. Model names then drop the "-cloud" suffix.

Cache key = sha256 of (model, messages incl. image bytes, temperature, seed).
A rerun of an experiment therefore makes zero API calls, and the cache is
what makes the LLM side reproducible (hosted models do not guarantee
determinism even with a fixed seed).
"""

import base64
import hashlib
import json
import os
import time
from pathlib import Path

import requests


class LLM:
    def __init__(self, model, temperature, cache_dir: Path):
        self.model = model
        self.temperature = temperature
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.api_key = os.environ.get("OLLAMA_API_KEY")

    def key(self, messages, seed):
        h = hashlib.sha256()
        h.update(json.dumps([self.model, self.temperature, seed]).encode())
        for m in messages:
            h.update(m["role"].encode() + m["content"].encode())
            for img in m.get("images", []):
                h.update(Path(img).read_bytes())
        return h.hexdigest()

    def chat(self, messages, seed):
        """messages: [{"role", "content", "images": [png paths]}]. Returns a dict
        with the answer text, token counts and whether it came from the cache."""
        key = self.key(messages, seed)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            out = json.loads(path.read_text())
            out["cached"] = True
            return out

        payload = {
            "model": self.model,
            "messages": [
                {"role": m["role"], "content": m["content"],
                 **({"images": [base64.b64encode(Path(p).read_bytes()).decode() for p in m["images"]]}
                    if m.get("images") else {})}
                for m in messages
            ],
            "stream": False,
            "options": {"temperature": self.temperature, "seed": seed, "num_ctx": 32768},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = None
        for attempt in range(6):
            t0 = time.time()
            try:
                r = requests.post(f"{self.host}/api/chat", json=payload, headers=headers, timeout=600)
            except requests.RequestException as e:
                err = repr(e)
            else:
                if r.status_code == 200:
                    break
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 403, 404):
                    raise RuntimeError(err)
            wait = min(600, 15 * 2 ** attempt)
            print(f"LLM call failed ({err}), retrying in {wait}s", flush=True)
            time.sleep(wait)
        else:
            raise RuntimeError(f"LLM call failed after retries: {err}")

        resp = r.json()
        out = {
            "model": resp.get("model", self.model),
            "content": resp["message"]["content"],
            "thinking": resp["message"].get("thinking", ""),
            "prompt_tokens": resp.get("prompt_eval_count", 0),
            "completion_tokens": resp.get("eval_count", 0),
            "latency_s": time.time() - t0,
            "seed": seed,
            "temperature": self.temperature,
            "n_images": sum(len(m.get("images", [])) for m in messages),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path.write_text(json.dumps(out, indent=1))
        out["cached"] = False
        return out
