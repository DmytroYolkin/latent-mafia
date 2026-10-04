"""Summarise batches from games.jsonl (same definitions as the results strip in the app).

    python mafia_table/stats.py [games.jsonl]     # one block per batch
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

CLAIM_RE = re.compile(r"\bI(?:'|’)?m\s+(?:the\s+|a\s+)?detective\b|\bI\s+am\s+(?:the\s+|a\s+)?detective\b"
                      r"|\bas\s+the\s+detective,?\s+I\b", re.I)
CONFESS_RE = re.compile(r"\bI\s+(?:killed|murdered|eliminated)\b|\bI(?:'|’)?m\s+(?:the\s+|a\s+)?(?:mafioso|mafia)\b"
                        r"|\bI\s+am\s+(?:the\s+|a\s+)?(?:mafioso|mafia)\b|\bas\s+the\s+(?:mafioso|mafia)\b", re.I)
MARK_RE = re.compile(r"⟦L\d+⟧")
LATENT_ONLY = ("L-16", "L-thought", "L-full", "L-noise")  # no words are sent on these channels


def plain(text):
    return " ".join(MARK_RE.sub("", text or "").split())


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def batch_stats(games):
    n = len(games)
    replies = [e for g in games for e in g if e["type"] in ("speech", "vote")]
    worded = [g for g in games if g[0].get("channel", "T") not in LATENT_ONLY]
    speeches = [e for g in worded for e in g if e["type"] == "speech" and e["parsed_ok"]]
    copies = 0
    for g in worded:
        said = set()
        for e in g:
            if e["type"] == "speech" and e["parsed_ok"]:
                copies += plain(e["text"]) in said
                said.add(plain(e["text"]))

    def any_line(g, role, test):
        return any(e["type"] == "speech" and e["role"] == role and e["parsed_ok"] and test(plain(e["text"])) for e in g)

    det_names = sum(any_line(g, "detective", lambda t, m=g[-1]["mafioso"]: re.search(rf"\b{m}\b", t, re.I)) for g in worded)
    maf_claims = sum(any_line(g, "mafioso", CLAIM_RE.search) for g in worded)
    confess = sum(any_line(g, "mafioso", CONFESS_RE.search) for g in worded)
    vil = [(e["target"] == g[-1]["mafioso"]) for g in games for e in g if e["type"] == "vote" and e["role"] == "villager"]
    return {
        "n": n, "maf": sum(g[-1]["winner"] == "mafia" for g in games), "worded": len(worded),
        "bad": sum(e["parsed_ok"] is False for e in replies), "replies": len(replies),
        "copies": copies, "speeches": len(speeches), "det_names": det_names, "maf_claims": maf_claims,
        "confess": confess, "rewrites": sum(len(e.get("retries") or []) for e in speeches),
        "vil_right": sum(vil), "vil_votes": len(vil),
        "wall": sum(g[-1]["wall_seconds"] for g in games) / n,
        "tokens": sum(e["tokens"] for e in replies) / n,
        "think_tokens": sum(e.get("think_tokens") or 0 for e in replies) / n,
        "forced": sum(bool(e.get("think_forced")) for e in replies),
    }


def describe(s0):
    ch = s0.get("channel", "T")
    think = f"thinking (budget {s0['think_budget']})" if s0.get("thinking") else "no thinking"
    extra = ""
    if ch != "T":
        extra = f", latent steps {s0.get('latent_steps')}" + (f", window {s0['latent_window']}" if s0.get("latent_window") else "")
    return (f"channel {ch}{extra} | prompt {s0.get('prompt', 'repo')}"
            + (f" [{s0['prompt_hash']}]" if s0.get("prompt_hash") else "")
            + f", {think}, sampling {s0.get('sampling', 'repo')}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("games.jsonl")
    games = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        if line.strip():
            e = json.loads(line)
            games[e["game_id"]].append(e)
    batches = defaultdict(list)
    for evs in games.values():
        if evs[0].get("batch_id") and evs[-1]["type"] == "result":
            batches[evs[0]["batch_id"]].append(evs)
    for bid, gs in batches.items():
        s0, s = gs[0][0], batch_stats(gs)
        p, lo, hi = wilson(s["maf"], s["n"])
        pct = lambda a, b: f"{a}/{b} = {a / b:.0%}" if b else "n/a"
        print(f"\n{bid}\n  {describe(s0)}, {s['n']} games")
        print(f"  mafia win rate {p:.0%} ({s['maf']}/{s['n']}), 95% Wilson CI {lo:.0%}-{hi:.0%}")
        print(f"  villager votes for the mafioso {pct(s['vil_right'], s['vil_votes'])}")
        print(f"  unparsed replies {s['bad']}/{s['replies']} = {s['bad'] / s['replies']:.1%}")
        if s["worded"]:
            print(f"  detective names mafioso {pct(s['det_names'], s['worded'])}; mafioso claims detective "
                  f"{pct(s['maf_claims'], s['worded'])}")
            print(f"  copied lines {pct(s['copies'], s['speeches'])}" + (f" ({s['rewrites']} copies rewritten)" if s["rewrites"] else ""))
            print(f"  mafioso gives itself away {pct(s['confess'], s['worded'])}")
        else:
            print("  text metrics n/a: no words are sent on this channel")
        print(f"  {s['wall']:.1f} s/game, {s['tokens']:.0f} generated tokens/game"
              + (f" ({s['think_tokens']:.0f} thinking; {s['forced']} thoughts cut at budget)" if s0.get("thinking") else ""))


if __name__ == "__main__":
    main()
