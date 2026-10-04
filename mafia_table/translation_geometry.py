"""Why do LatentMAS (identity) latents fail and soft tokens work? Geometry of the saved latent vectors
against the model's real input embeddings (CPU only; reads logs/latents/*.npz and games.jsonl).

For each latent vector v:
  nearest-cos  cosine to the closest real token embedding (how word-like the input is)
  self-match   is v closest to the token it predicts? (the logit-lens token: argmax of E @ v, which
               equals the model's own next-token choice because input and output embeddings are tied)
  top-25 dims  share of v's squared norm held by its 25 largest coordinates (1% of 2560)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "llm-mafia-game" / "models" / "Qwen3-4B-Q4_K_M.gguf"


def main(limit=400):
    import gguf

    reader = gguf.GGUFReader(str(MODEL))
    te = next(t for t in reader.tensors if t.name == "token_embd.weight")
    E = gguf.quants.dequantize(te.data, te.tensor_type).astype(np.float32).reshape(-1, te.shape[0])
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    translation = {}
    for line in (HERE / "games.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        if e["type"] == "start" and e.get("channel") in ("L-16", "L-thought", "L-full", "H"):
            translation[e["game_id"]] = e.get("latent_translation") or "identity"
    vecs = defaultdict(list)
    for f in sorted((HERE / "logs" / "latents").glob("*.npz")):
        t = translation.get(f.stem)
        if not t:
            continue
        z = np.load(f)
        for k in z.files:
            if k.startswith("L"):
                vecs[t].extend(z[k].astype(np.float32))
    rng = np.random.default_rng(0)
    rows = {"real token embeddings": E[rng.choice(len(E), limit, replace=False)]}
    for t in ("identity", "soft"):
        if vecs[t]:
            v = np.array(vecs[t])
            rows[f"{t} latents"] = v[rng.choice(len(v), min(limit, len(v)), replace=False)]
    print(f"{'vectors':24} {'n':>4} {'nearest-cos':>12} {'self-match':>11} {'top-25 dims':>12} {'norm':>6}")
    for name, V in rows.items():
        Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
        sims = Vn @ En.T                                # cosine to every token embedding
        nearest = sims.max(1)
        if name.startswith("real"):                     # a real embedding's nearest is itself; take the 2nd
            nearest = np.sort(sims, axis=1)[:, -2]
        predicted = (V @ E.T).argmax(1)                  # logit-lens token
        self_match = (sims.argmax(1) == predicted).mean()
        sq = V ** 2
        top = np.sort(sq, axis=1)[:, -25:].sum(1) / sq.sum(1)
        print(f"{name:24} {len(V):4d} {nearest.mean():12.2f} {self_match:11.0%} {top.mean():12.0%} "
              f"{np.linalg.norm(V, axis=1).mean():6.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
