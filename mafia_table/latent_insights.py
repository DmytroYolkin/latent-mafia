"""What do the latent channels change? A report from games.jsonl and logs/latents/*.npz.

    llm-mafia-game\\.venv\\Scripts\\python mafia_table\\latent_insights.py [--since 2026-10-04]

1. Outcomes per channel: mafia win rate, and the villager's vote accuracy (the cleanest readout of what
   reached the villager, because it works on every channel).
2. Paired contrasts (Fisher exact test on villager accuracy):
   L-16 vs L-noise   is there usable content in the latent vectors, or are listeners only perturbed?
   L-full vs L-thought  does handing over the speaker's prompt (its secret role) leak the role?
   H vs T            do the vectors add anything on top of the words?
   L-16 vs T         latent vectors instead of words
3. Role probe: can a linear read-out recover the speaker's role from its own latent vectors? High
   accuracy with low villager accuracy = the private information is in the vectors, but the listening
   model cannot use it.
4. Logit lens: how word-like the latent vectors are (share of lens tokens that are words).
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LATENT_DIR = HERE / "logs" / "latents"
ORDER = ["T", "H", "L-16", "L-noise", "L-thought", "L-full",
         "H soft", "L-16 soft", "L-noise soft", "L-thought soft", "L-full soft"]


def key(start):
    """Channel plus translation: soft-token runs are reported as separate rows."""
    ch = start.get("channel", "T")
    return ch + (" soft" if start.get("latent_translation") == "soft" else "")


def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fisher_two_sided(a, b, c, d):
    """2x2 table [[a, b], [c, d]]: exact two-sided p-value."""
    n1, n2, k = a + b, c + d, a + c
    pmf = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
    p0 = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pmf(x) <= p0 * (1 + 1e-9)))


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial test of k successes in n against probability p."""
    pmf = lambda i: math.comb(n, i) * p ** i * (1 - p) ** (n - i)
    pk = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= pk * (1 + 1e-9)))


def coinflip_mafia_rate(gs):
    """Mafia win rate if each game's villager had voted at random, with the detective's and mafioso's
    actual votes kept. On a channel where nothing usable reaches the villager, observed ~= this."""
    total = 0.0
    for g in gs:
        votes = {e["speaker"]: e["target"] for e in g if e["type"] == "vote"}
        vil = next(e["speaker"] for e in g if e["type"] == "vote" and e["role"] == "villager")
        for choice in [p for p in votes if p != vil]:
            v = dict(votes)
            v[vil] = choice
            c = Counter(v.values())
            top = max(c.values())
            tied = [t for t in c if c[t] == top]
            total += 0.5 * sum(g[-1]["roles"][t] != "mafioso" for t in tied) / len(tied)
    return total / len(gs)


def load_games(path, since):
    games = defaultdict(list)
    for line in Path(path).open(encoding="utf-8"):
        if line.strip():
            e = json.loads(line)
            games[e["game_id"]].append(e)
    out = []
    for gid, evs in games.items():
        s0 = evs[0]
        if evs[-1]["type"] != "result" or s0.get("mock") or not s0.get("batch_id"):
            continue
        if since and s0["game_id"] < since.replace("-", ""):
            continue
        # comparable settings only: the fair prompt, no thinking, Qwen sampling
        if s0.get("prompt") != "fair" or s0.get("thinking") or s0.get("sampling") != "qwen":
            continue
        if not s0.get("prompt_hash"):  # fair v1 (before the "To win" fix) recorded no hash
            continue
        out.append(evs)
    return out


def outcomes(games):
    by = defaultdict(list)
    for g in games:
        by[key(g[0])].append(g)
    rows = {}
    for ch, gs in by.items():
        maf = sum(g[-1]["winner"] == "mafia" for g in gs)
        vil = [e["target"] == g[-1]["mafioso"] for g in gs for e in g if e["type"] == "vote" and e["role"] == "villager"]
        bad = sum(e.get("parsed_ok") is False for g in gs for e in g if e["type"] in ("speech", "vote"))
        rep = sum(1 for g in gs for e in g if e["type"] in ("speech", "vote"))
        rows[ch] = {"n": len(gs), "maf": maf, "vil_right": sum(vil), "vil_n": len(vil), "bad": bad, "replies": rep,
                    "sec": sum(g[-1]["wall_seconds"] for g in gs) / len(gs),
                    "p_coin": binom_two_sided(sum(vil), len(vil)), "maf_if_guess": coinflip_mafia_rate(gs)}
    return rows


def ridge_probe(X, y, grp, lam=10.0):
    """Leave-one-game-out ridge classifier (one-vs-rest least squares on standardised features).
    Returns (3-way balanced accuracy, mafioso-vs-rest balanced accuracy); chance = 1/3 and 1/2."""
    labels = sorted(set(y))
    pred = np.empty(len(y), dtype=object)
    for gid in sorted(set(grp)):
        te, tr = grp == gid, grp != gid
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
        Y = np.array([[1.0 if v == lab else -1.0 for lab in labels] for v in y[tr]])
        # dual form: (A A^T + lam I)^-1 Y, since features (2560) >> samples (~180)
        alpha = np.linalg.solve(A @ A.T + lam * np.eye(len(A)), Y)
        scores = B @ (A.T @ alpha)
        pred[te] = np.array(labels)[scores.argmax(1)]
    bal = np.mean([(pred[y == lab] == lab).mean() for lab in labels])
    is_m, pm = (y == "mafioso"), (pred == "mafioso")
    bal_m = 0.5 * ((pm[is_m]).mean() + (~pm[~is_m]).mean()) if is_m.any() and (~is_m).any() else float("nan")
    return bal, bal_m


def probe(games):
    """Can a linear read-out recover the speaker's role from its latent vectors? Leave-one-game-out
    ridge probe, per channel, on the mean of the 16 vectors and on the first vector alone.
    L-noise is the sanity control (should be at chance)."""
    X, X1, y, grp, chan = [], [], [], [], []
    for g in games:
        f = LATENT_DIR / f"{g[0]['game_id']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        for k, m in meta.items():
            v = z[f"L{k}"].astype(np.float32)
            if len(v):
                X.append(v.mean(axis=0))
                X1.append(v[0])
                y.append(m["role"])
                grp.append(g[0]["game_id"])
                chan.append(key(g[0]))
    if not X:
        return {}
    X, X1, y, grp, chan = np.array(X), np.array(X1), np.array(y), np.array(grp), np.array(chan)
    res = {}
    for ch in sorted(set(chan), key=ORDER.index):
        m = chan == ch
        mean3, mean_m = ridge_probe(X[m], y[m], grp[m])
        first3, first_m = ridge_probe(X1[m], y[m], grp[m])
        res[ch] = {"messages": int(m.sum()), "mean3": mean3, "mean_m": mean_m, "first3": first3, "first_m": first_m}
    return res


def lens_summary(games):
    word = re.compile(r"^\s?[A-Za-z]{2,}$")
    by = defaultdict(Counter)
    share = defaultdict(lambda: [0, 0])
    for g in games:
        for e in g:
            lat = e.get("latent") if e["type"] == "speech" else None
            if not lat or lat.get("noise"):
                continue
            for top in lat.get("lens") or []:
                t = top[0] if top else ""
                by[key(g[0])][t] += 1
                share[key(g[0])][0] += bool(word.match(t))
                share[key(g[0])][1] += 1
    return by, share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(HERE / "games.jsonl"))
    ap.add_argument("--since", default=None, help="only games from this date on, e.g. 2026-10-04")
    args = ap.parse_args()
    games = load_games(args.log, args.since)
    rows = outcomes(games)
    lines = ["# Latent channel insights", "",
             "Settings compared: fair prompt, no thinking, Qwen sampling (T games with the same settings are the baseline).", "",
             "## 1. Outcomes per channel", "",
             "The villager is the swing vote: the detective votes for the mafioso from its own investigation, the",
             "mafioso votes for someone else. So read a channel by the villager's accuracy against a coin flip, and",
             "the mafia win rate against what a guessing villager would give.", "",
             "| Channel | Games | Villager votes for the mafioso (95% CI) | p vs coin flip | Mafia wins (95% CI) | Mafia wins if the villager guessed | Unparsed | s / game |",
             "|---|---|---|---|---|---|---|---|"]
    for ch in [c for c in ORDER if c in rows]:
        r = rows[ch]
        p, lo, hi = wilson(r["maf"], r["n"])
        q, qlo, qhi = wilson(r["vil_right"], r["vil_n"])
        lines.append(f"| {ch} | {r['n']} | {q:.0%} ({qlo:.0%}–{qhi:.0%}) | {r['p_coin']:.3f} | {p:.0%} ({lo:.0%}–{hi:.0%}) | "
                     f"{r['maf_if_guess']:.0%} | {r['bad'] / r['replies']:.1%} | {r['sec']:.0f} |")
    lines += ["", "## 2. Paired contrasts (villager accuracy, Fisher exact test)", ""]
    for a, b, question in [("L-16 soft", "L-16", "does the translation matter? (soft tokens vs LatentMAS identity)"),
                           ("L-16 soft", "L-noise soft", "is there usable content in soft latents?"),
                           ("L-16 soft", "T", "soft latent vectors instead of words"),
                           ("L-full soft", "L-thought soft", "with soft latents: does sharing the prompt add a leak?"),
                           ("L-16", "L-noise", "is there usable content in the latents?"),
                           ("L-full", "L-thought", "does sharing the speaker's prompt leak its role?"),
                           ("H", "T", "do vectors add anything to words?"),
                           ("L-16", "T", "latent vectors instead of words")]:
        if a in rows and b in rows:
            ra, rb = rows[a], rows[b]
            pv = fisher_two_sided(ra["vil_right"], ra["vil_n"] - ra["vil_right"], rb["vil_right"], rb["vil_n"] - rb["vil_right"])
            lines.append(f"- **{a} vs {b}** ({question}): {ra['vil_right']}/{ra['vil_n']} vs {rb['vil_right']}/{rb['vil_n']}, p = {pv:.3f}")
        else:
            lines.append(f"- {a} vs {b}: not enough data yet")
    pr = probe(games)
    lines += ["", "## 3. Role probe on the latent vectors (leave-one-game-out ridge, balanced accuracy)", "",
              "Chance: 33% for the 3 roles, 50% for mafioso vs rest.", "",
              "| Channel | Messages | 3 roles, mean vector | Mafioso vs rest, mean vector | 3 roles, first vector | Mafioso vs rest, first vector |",
              "|---|---|---|---|---|---|"]
    for ch, r in pr.items():
        lines.append(f"| {ch} | {r['messages']} | {r['mean3']:.0%} | {r['mean_m']:.0%} | {r['first3']:.0%} | {r['first_m']:.0%} |")
    by, share = lens_summary(games)
    lines += ["", "## 4. Logit lens of the latent vectors", ""]
    for ch in [c for c in ORDER if c in by]:
        w, n = share[ch]
        top = ", ".join(f"{t!r} {c}" for t, c in by[ch].most_common(6))
        lines.append(f"- {ch}: {w / n:.0%} of lens tokens are words; most common: {top}")
    report = "\n".join(lines)
    print(report)
    out = HERE / "logs" / "latent_insights.md"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\n(saved to {out})")


if __name__ == "__main__":
    main()
