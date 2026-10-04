"""Latent channels for the game table (LatentMAS-style message passing on llama.cpp).

Channel    what a listener receives from a speaker
T          the spoken text (the default game; not handled here)
L-16       the speaker's 16 latent vectors, placed in the listener's own context where the message goes
L-thought  only the KV cache of the speaker's latent steps, in one cache shared by the game
           (LatentMAS "latent_only": each speaker's prompt is removed from the shared cache)
L-full     the speaker's whole KV cache, prompt with its secret role included, plus its latent steps,
           in one cache shared by the game (LatentMAS default handoff)
H          the spoken text plus the 16 latent vectors
L-noise    control: 16 random vectors with the same norm as real latents, in place of L-16's vectors

A latent step (LatentMAS models.py:321-348): take the last hidden state, rescale it to the mean L2 norm
of the input-embedding table, and feed it back as an input embedding. Qwen3-4B ties its input and
output embeddings, so LatentMAS's realignment matrix is about the identity and the rescale is all of it.

Votes stay text. For viewers only, every latent speaker also writes what it would have said (the
gloss); the gloss is never sent to other players.
"""
import ctypes
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from llm import CHAT_PREFIX, CHAT_SUFFIX, SAMPLING_PRESETS

CHANNELS = ("T", "L-16", "L-thought", "L-full", "H", "L-noise")
SHARED_CHANNELS = ("L-thought", "L-full")       # one growing KV cache per game
VECTOR_CHANNELS = ("L-16", "H", "L-noise")      # vectors placed in each listener's own context
TEXT_CHANNELS = ("T", "H")                      # the message includes words
MARK_RE = re.compile(r"⟦L(\d+)⟧")
SHARED_ANCHOR = ("<|im_start|>system\nShared memory of a Mini-Mafia game. The players' turns follow "
                 "in order.<|im_end|>\n")
LATENT_CTX = 12288   # L-full keeps every speaker's whole prompt: about 6 x 1400 tokens + a vote
N_BATCH = 512


def strip_marks(text):
    return " ".join(MARK_RE.sub("", text or "").split())


def embedding_norm(model_path, cache_file):
    """Mean L2 norm of the input-embedding table. Dequantizing the Q6_K table takes ~25 s, so cache it."""
    st = os.stat(model_path)
    key = f"{os.path.basename(model_path)}:{st.st_size}"
    try:
        d = json.loads(Path(cache_file).read_text())
        if d.get("key") == key:
            return d["mean_norm"]
    except (OSError, ValueError):
        pass
    import gguf

    reader = gguf.GGUFReader(model_path)
    te = next(t for t in reader.tensors if t.name == "token_embd.weight")
    W = gguf.quants.dequantize(te.data, te.tensor_type).astype(np.float32).reshape(-1, te.shape[0])
    norm = float(np.linalg.norm(W, axis=1).mean())
    Path(cache_file).write_text(json.dumps({"key": key, "mean_norm": norm}))
    return norm


def sample(logits, p, recent, rng):
    """llama.cpp's sampler order: penalties -> top_k -> top_p -> min_p -> temperature -> draw."""
    x = logits.astype(np.float64)
    if recent:
        idx = np.fromiter(set(recent[-64:]), dtype=np.int64)
        rp = p.get("repeat_penalty", 1.0)
        if rp != 1.0:
            x[idx] = np.where(x[idx] > 0, x[idx] / rp, x[idx] * rp)
        x[idx] -= p.get("presence_penalty", 0.0)
    k = p.get("top_k", 40)
    cand = np.argpartition(-x, k)[:k] if 0 < k < x.size else np.arange(x.size)
    cand = cand[np.argsort(-x[cand])]
    v = x[cand]
    pr = np.exp(v - v.max())
    pr /= pr.sum()
    keep = (np.cumsum(pr) - pr) < p.get("top_p", 1.0)
    if p.get("min_p", 0.0) > 0:
        keep &= pr >= p["min_p"] * pr[0]
    cand, v = cand[keep], v[keep] / max(p["temp"], 1e-5)
    pr = np.exp(v - v.max())
    return int(rng.choice(cand, p=pr / pr.sum()))


class LatentEngine:
    """A second llama.cpp context on the already-loaded model (no second copy of the weights):
    two sequences (0 = main, 1 = fork), unified KV cache, per-token hidden states on demand."""

    def __init__(self, backend, cache_dir, n_ctx=LATENT_CTX):
        import llama_cpp as L

        self.L, self.b, llm = L, backend, backend.llm
        p = L.llama_context_default_params()
        p.n_ctx, p.n_batch, p.n_ubatch, p.n_seq_max, p.kv_unified = n_ctx, N_BATCH, N_BATCH, 2, True
        p.embeddings, p.pooling_type = False, L.LLAMA_POOLING_TYPE_NONE
        p.n_threads = p.n_threads_batch = max(1, (os.cpu_count() or 4) // 2)
        self.ctx = L.llama_init_from_model(llm._model.model, p)
        if not self.ctx:
            raise RuntimeError("could not create the latent llama.cpp context (out of memory?)")
        self.mem = L.llama_get_memory(self.ctx)
        self.n_ctx = n_ctx
        self.n_embd = L.llama_model_n_embd(llm._model.model)
        self.n_vocab = llm.n_vocab()
        self.mean_norm = embedding_norm(backend.model_path, Path(cache_dir) / "embedding_norm.json")
        self.stop_ids = backend.stop_ids
        self.pos = {0: 0, 1: 0}
        self.max_used = 0
        self._E = None  # input-embedding table (float32), loaded on first soft-token use

    def embedding_table(self):
        """The dequantized input-embedding table (151936 x 2560, ~1.5 GB of RAM), for soft tokens."""
        if self._E is None:
            import gguf

            reader = gguf.GGUFReader(self.b.model_path)
            te = next(t for t in reader.tensors if t.name == "token_embd.weight")
            self._E = gguf.quants.dequantize(te.data, te.tensor_type).astype(np.float32).reshape(-1, te.shape[0])
        return self._E

    def translate(self, h, logits, translation):
        """Hidden state -> next input vector.
        identity  LatentMAS: h rescaled to the mean input-embedding norm. Measured on Qwen3-4B: these
                  vectors sit far from every real token embedding (cosine ~0.15) and a fresh listener
                  reads them like noise (translation_test.py).
        soft      soft token: sum_t softmax(logits)_t * E[t] over the top 256 tokens (renormalised).
                  Stays inside the space of real embeddings (cosine ~0.96); the listener recovers the
                  speaker's secret 60/60 in translation_test.py."""
        if translation == "soft":
            top = np.argpartition(-logits, 256)[:256]
            p = np.exp(logits[top] - logits[top].max())
            return (p / p.sum()) @ self.embedding_table()[top]
        return h / (np.linalg.norm(h) + 1e-6) * self.mean_norm

    # ---- low level -----------------------------------------------------------------------
    def tok(self, text):
        return self.b.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)

    def detok(self, ids):
        return self.b.llm.detokenize(ids).decode("utf-8", errors="ignore")

    def reset(self):
        self.L.llama_memory_clear(self.mem, True)
        self.pos = {0: 0, 1: 0}

    def fork(self):
        """seq 1 := seq 0 (cells are shared, nothing is copied)."""
        self.L.llama_memory_seq_rm(self.mem, 1, -1, -1)
        self.L.llama_memory_seq_cp(self.mem, 0, 1, -1, -1)
        self.pos[1] = self.pos[0]

    def drop(self):
        self.L.llama_memory_seq_rm(self.mem, 1, -1, -1)

    def remove(self, seq, p0, p1):
        self.L.llama_memory_seq_rm(self.mem, seq, p0, p1)

    def _decode(self, seq, tokens=None, embs=None, last_out=True, hidden=False):
        L = self.L
        n = len(tokens) if tokens is not None else len(embs)
        if self.pos[seq] + n > self.n_ctx:
            from llm import ContextOverflow
            raise ContextOverflow(f"latent context needs {self.pos[seq] + n} positions > {self.n_ctx}")
        L.llama_set_embeddings(self.ctx, hidden)
        done = 0
        while done < n:
            k = min(N_BATCH, n - done)
            b = L.llama_batch_init(k, 0 if tokens is not None else self.n_embd, 1)
            try:
                if embs is not None:
                    arr = np.ascontiguousarray(embs[done:done + k], dtype=np.float32).ravel()
                    ctypes.memmove(b.embd, arr.ctypes.data, arr.nbytes)
                for i in range(k):
                    if tokens is not None:
                        b.token[i] = tokens[done + i]
                    b.pos[i], b.n_seq_id[i] = self.pos[seq] + i, 1
                    b.seq_id[i][0] = seq
                    b.logits[i] = 1 if (last_out and done + i == n - 1) else 0
                b.n_tokens = k
                rc = L.llama_decode(self.ctx, b)
            finally:
                L.llama_batch_free(b)
            if rc != 0:
                raise RuntimeError(f"llama_decode returned {rc} (latent context)")
            self.pos[seq] += k
            done += k
        self.max_used = max(self.max_used, self.pos[seq])

    def logits(self):
        return np.ctypeslib.as_array(self.L.llama_get_logits_ith(self.ctx, -1), shape=(self.n_vocab,)).copy()

    def hidden(self):
        return np.ctypeslib.as_array(self.L.llama_get_embeddings_ith(self.ctx, -1), shape=(self.n_embd,)).copy()

    # ---- building blocks -----------------------------------------------------------------
    def feed(self, seq, segments):
        """Decode [("t", text) | ("e", vectors)] segments; return (logits, hidden) at the last position.
        Embedding output is switched on only for the last position (otherwise llama.cpp would compute
        outputs for every token of the prompt)."""
        items = []
        for kind, val in segments:
            if kind == "t":
                ids = self.tok(val)
                if ids:
                    items.append(("t", ids))
            elif len(val):
                items.append(("e", np.asarray(val, dtype=np.float32)))
        kind, val = items[-1]
        head, last = (items[:-1] + [(kind, val[:-1])]), (kind, val[-1:])
        for k2, v2 in head:
            if len(v2):
                self._decode(seq, tokens=v2 if k2 == "t" else None, embs=v2 if k2 == "e" else None, last_out=False)
        k2, v2 = last
        self._decode(seq, tokens=v2 if k2 == "t" else None, embs=v2 if k2 == "e" else None, hidden=True)
        return self.logits(), self.hidden()

    def latent_steps(self, seq, h, logits, m, lens_k=3, translation="identity"):
        """m latent steps from hidden state h (LatentMAS loop; `translation` picks how a hidden state
        becomes the next input). Returns the fed vectors and, for each, the top tokens of the output
        layer at the position that produced it (the logit lens of that vector)."""
        vecs, lens = [], []
        for _ in range(m):
            lens.append([self.detok([int(t)]) for t in np.argsort(-logits)[:lens_k]])
            e = self.translate(h, logits, translation)
            vecs.append(e)
            self._decode(seq, embs=e[None, :], hidden=True)
            logits, h = self.logits(), self.hidden()
        return np.array(vecs, dtype=np.float32).reshape(-1, self.n_embd), lens

    def generate(self, seq, logits, budget, params, rng, stop_fn=None, force=None, on_token=None):
        """Sample text in `seq`, starting from `logits`. Same budget / stop / forced-line rules as
        LlamaBackend.generate."""
        toks, shown = [], ""

        def run(lg, n):
            nonlocal shown
            for _ in range(n):
                t = sample(lg, params, toks, rng)
                if t in self.stop_ids:
                    return
                toks.append(t)
                self._decode(seq, tokens=[t])
                lg = self.logits()
                text = self.detok(toks)
                if on_token and len(text) > len(shown):
                    on_token(text[len(shown):], "answer")
                    shown = text
                if stop_fn and stop_fn(text):
                    return

        run(logits, budget)
        forced = False
        if force:
            m = force["marker"].search(self.detok(toks).replace("\\n", "\n"))
            if not m or not m.group(1).strip():
                forced = True
                inject = self.tok(force["inject"] if not m else " ")
                toks.extend(inject)
                self._decode(seq, tokens=inject)
                run(self.logits(), force["budget"])
        return self.detok(toks).strip(), len(toks), forced


class ChannelRunner:
    """One game on a latent channel: turns the repo engine's generate() calls into channel traffic.
    Messages travel through markers: a latent speaker's reply becomes 'SAY: ⟦L7⟧', the repo engine
    stores 'Bob: "⟦L7⟧"' in every listener's memory, and later prompts expand the marker again."""

    def __init__(self, engine, channel, steps, sampling, seed, game_id, save_dir, text_backend=None, window=3,
                 translation="identity"):
        self.e, self.channel, self.steps, self.sampling = engine, channel, steps, sampling
        self.translation = translation
        self.rng = np.random.default_rng(seed)
        self.game_id, self.save_dir = game_id, Path(save_dir)
        self.text_backend = text_backend          # mock mode: no engine, text from the mock backend
        self.store, self.n = {}, 0
        self.open_turn = False  # shared cache ends with latent cells of an unclosed assistant turn
        # Shared channels keep only the last `window` turns (one per surviving player by default).
        # Measured: with 4+ latent turns in one cache, Qwen3-4B can no longer start a normal reply (the
        # vote opens with " the" and loops), while 3 turns work; ordinary text or word embeddings in the
        # same places do not cause this. LatentMAS's own chain has 3 latent agents. window=None keeps all.
        self.window, self.turns = window, []  # turns: (first, last+1) cache positions of each kept turn
        if engine:
            engine.reset()
            if self.shared:
                # A fixed opening that is never removed. Without it, L-thought's cache would start with
                # bare latent cells, and models lean heavily on the first positions (attention sink).
                engine.feed(0, [("t", SHARED_ANCHOR)])

    @property
    def shared(self):
        return self.channel in SHARED_CHANNELS

    def _segments(self, prompt):
        """Chat-wrapped prompt as segments; markers become vectors (vector channels) or a short note
        (shared channels: the content is already in the shared cache)."""
        text = CHAT_PREFIX + prompt + CHAT_SUFFIX
        if self.shared:
            return [("t", MARK_RE.sub("(latent message, in the shared memory above)", text))]
        segs, last = [], 0
        for m in MARK_RE.finditer(text):
            segs.append(("t", text[last:m.start()] + "(latent message) "))
            segs.append(("e", self.store[int(m.group(1))]["vectors"]))
            last = m.end()
        segs.append(("t", text[last:]))
        return segs

    def _noise(self, m, norm=None):
        """Random vectors; `norm` = the mean norm of the real latents they replace (same translation)."""
        v = self.rng.standard_normal((m, self.e.n_embd if self.e else 2560)).astype(np.float32)
        norm = norm or (self.e.mean_norm if self.e else 1.0)
        return v / np.linalg.norm(v, axis=1, keepdims=True) * norm

    def reply(self, agent, prompt, kind, budget, params, stop_fn, force, on_token):
        t0 = time.perf_counter()
        if kind == "vote":
            text, ntok, forced, ptok = self._vote(agent, prompt, budget, params, stop_fn, force, on_token)
            return {"text": text, "tokens": ntok, "prompt_tokens": ptok, "seconds": time.perf_counter() - t0,
                    "line_forced": forced, "latent": None}
        return self._speech(agent, prompt, budget, params, stop_fn, force, on_token, t0)

    def _vote(self, agent, prompt, budget, params, stop_fn, force, on_token):
        if not self.e:  # mock
            r = self.text_backend.generate(MARK_RE.sub("(latent message)", prompt), budget, agent=agent,
                                           stop_fn=stop_fn, force=force, on_token=on_token)
            return r["text"], r["tokens"], False, r["prompt_tokens"]
        if self.shared:
            self.e.fork()
            start = self.e.pos[1]
            lg, _ = self.e.feed(1, [("t", "<|im_end|>\n" if self.open_turn else "")] + self._segments(prompt))
            ptok = self.e.pos[1] - start
            text, ntok, forced = self.e.generate(1, lg, budget, params, self.rng, stop_fn, force, on_token)
            self.e.drop()
        else:
            self.e.reset()
            lg, _ = self.e.feed(0, self._segments(prompt))
            ptok = self.e.pos[0]
            text, ntok, forced = self.e.generate(0, lg, budget, params, self.rng, stop_fn, force, on_token)
        return text, ntok, forced, ptok

    def _speech(self, agent, prompt, budget, params, stop_fn, force, on_token, t0):
        self.n += 1
        lid = self.n
        if not self.e:  # mock: text from the mock backend, random "latents"
            r = self.text_backend.generate(MARK_RE.sub("(latent message)", prompt), budget, agent=agent,
                                           stop_fn=stop_fn, force=force, on_token=on_token)
            words, ntok, ptok = r["text"], r["tokens"], r["prompt_tokens"]
            vecs, lens = self._noise(self.steps), [["(mock)"]] * self.steps
        else:
            if self.shared:
                if self.window:
                    while len(self.turns) >= self.window:  # make room: forget the oldest turn
                        p0, p1 = self.turns.pop(0)
                        self.e.remove(0, p0, p1)
                start = self.e.pos[0]
                lg, h = self.e.feed(0, [("t", "<|im_end|>\n" if self.open_turn else "")] + self._segments(prompt))
            else:
                self.e.reset()
                start = 0
                lg, h = self.e.feed(0, self._segments(prompt))
            prompt_end = self.e.pos[0]
            ptok = prompt_end - start
            # what it would have said (gloss), or the real text for H; from a fork, before any latent step
            self.e.fork()
            words, ntok, _ = self.e.generate(1, lg, budget, params, self.rng, stop_fn, force, on_token)
            self.e.drop()
            vecs, lens = self.e.latent_steps(0, h, lg, self.steps, translation=self.translation)
            if self.channel == "L-noise":
                real_norm = float(np.linalg.norm(vecs, axis=1).mean()) if len(vecs) else None
                vecs, lens = self._noise(self.steps, real_norm), [["(noise)"]] * self.steps
            if self.channel == "L-thought":
                self.e.remove(0, start, prompt_end)  # LatentMAS latent_only: keep only the latent steps
            if self.shared:
                self.turns.append((prompt_end if self.channel == "L-thought" else start, self.e.pos[0]))
            self.open_turn = self.shared  # the speaker's assistant turn is left open after its latents
        self.store[lid] = {"vectors": vecs, "speaker": agent.name, "role": agent.role}
        import speech
        notes, said = speech.split_reply(words)
        said = speech.parse_briefed_say(words) if said else None
        if self.channel == "H":
            reply = f"PRIVATE: {notes}\nSAY: {(said + ' ') if said else ''}⟦L{lid}⟧\n"
        else:
            reply = f"SAY: ⟦L{lid}⟧\n"
        return {"text": reply, "tokens": ntok + self.steps, "prompt_tokens": ptok,
                "seconds": time.perf_counter() - t0, "line_forced": False,
                "latent": {"id": lid, "steps": self.steps, "lens": lens, "noise": self.channel == "L-noise",
                           "translation": self.translation,
                           "gloss": None if self.channel == "H" else said, "gloss_notes": notes or None,
                           "norm": float(np.linalg.norm(vecs, axis=1).mean()) if len(vecs) else 0.0}}

    def save(self):
        """Latent vectors of every message, for offline probes (logs/latents/<game_id>.npz)."""
        if not self.store:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.save_dir / f"{self.game_id}.npz",
                            **{f"L{k}": v["vectors"].astype(np.float16) for k, v in self.store.items()},
                            meta=json.dumps({k: {"speaker": v["speaker"], "role": v["role"]}
                                             for k, v in self.store.items()}))
