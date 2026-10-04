"""Is the latent "translation" the problem? A referential message-passing test, outside the game.

A speaker is told a secret ("<name> is the mafioso") and sends a message of m vectors; a fresh listener
sees "Bob sent you this message: <message>" and must name the mafioso among 3 candidates (chance 33%).
The speaker produces its m vectors with LatentMAS-style steps, but with different ways of turning a
hidden state h into the next input vector (the "translation"):

  identity     h rescaled to the mean input-embedding norm (what LatentMAS does; Qwen3-4B ties its
               embeddings, so LatentMAS's realignment matrix is ~identity too)
  soft-1.0     soft token: sum_t softmax(logits/1.0)_t * E[t]  (probability-weighted real embeddings)
  soft-0.5     same, sharper
  hard         E[argmax]: the embedding of the greedy token (text, delivered as embeddings)
  noise        random vectors with the same norm (control)
Plus controls: "text" (the speaker's greedy words, as ordinary tokens) and "none" (no message).

    llm-mafia-game\\.venv\\Scripts\\python mafia_table\\translation_test.py [--trials 60] [--steps 16]
"""
import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from llm import add_cuda_dll_dirs  # noqa: E402

add_cuda_dll_dirs()
import gguf  # noqa: E402
import llama_cpp as L  # noqa: E402
from llama_cpp import Llama  # noqa: E402

MODEL = HERE.parent / "llm-mafia-game" / "models" / "Qwen3-4B-Q4_K_M.gguf"
NAMES = ["Alice", "Charlie", "Diana"]
NL = chr(10)
U, A = "<|im_start|>user" + NL, "<|im_end|>" + NL + "<|im_start|>assistant" + NL + "<think>" + NL + NL + "</think>" + NL + NL


class Engine:
    def __init__(self):
        reader = gguf.GGUFReader(str(MODEL))
        te = next(t for t in reader.tensors if t.name == "token_embd.weight")
        self.E = gguf.quants.dequantize(te.data, te.tensor_type).astype(np.float32).reshape(-1, te.shape[0])
        self.mean_norm = float(np.linalg.norm(self.E, axis=1).mean())
        self.llm = Llama(model_path=str(MODEL), n_gpu_layers=-1, n_ctx=2048, verbose=False, embedding=False,
                         pooling_type=L.LLAMA_POOLING_TYPE_NONE)
        self.ctx, self.mem = self.llm._ctx.ctx, L.llama_get_memory(self.llm._ctx.ctx)
        self.n_embd, self.n_vocab, self.pos = self.E.shape[1], self.llm.n_vocab(), 0

    def tok(self, s):
        return self.llm.tokenize(s.encode(), add_bos=False, special=True)

    def reset(self):
        L.llama_memory_clear(self.mem, True)
        self.pos = 0

    def decode(self, tokens=None, embs=None, hidden=False):
        n = len(tokens) if tokens is not None else len(embs)
        L.llama_set_embeddings(self.ctx, hidden)
        b = L.llama_batch_init(n, 0 if tokens is not None else self.n_embd, 1)
        if embs is not None:
            arr = np.ascontiguousarray(embs, dtype=np.float32).ravel()
            ctypes.memmove(b.embd, arr.ctypes.data, arr.nbytes)
        for i in range(n):
            if tokens is not None:
                b.token[i] = tokens[i]
            b.pos[i], b.n_seq_id[i], b.logits[i] = self.pos + i, 1, 1 if i == n - 1 else 0
            b.seq_id[i][0] = 0
        b.n_tokens = n
        assert L.llama_decode(self.ctx, b) == 0
        L.llama_batch_free(b)
        self.pos += n

    def logits(self):
        return np.ctypeslib.as_array(L.llama_get_logits_ith(self.ctx, -1), shape=(self.n_vocab,)).copy()

    def hidden(self):
        return np.ctypeslib.as_array(L.llama_get_embeddings_ith(self.ctx, -1), shape=(self.n_embd,)).copy()

    def feed_prompt_with_hidden(self, text):
        ids = self.tok(text)
        self.decode(tokens=ids[:-1])
        self.decode(tokens=ids[-1:], hidden=True)
        return self.logits(), self.hidden()

    def translate(self, kind, h, logits, rng):
        if kind == "identity":
            return h / (np.linalg.norm(h) + 1e-6) * self.mean_norm
        if kind.startswith("soft"):
            tau = float(kind.split("-")[1])
            top = np.argpartition(-logits, 256)[:256]
            p = np.exp((logits[top] - logits[top].max()) / tau)
            p /= p.sum()
            return p @ self.E[top]
        if kind == "hard":
            return self.E[int(np.argmax(logits))]
        if kind == "noise":
            v = rng.standard_normal(self.n_embd).astype(np.float32)
            return v / np.linalg.norm(v) * self.mean_norm
        raise ValueError(kind)

    def speak(self, secret, kind, steps, rng):
        """Speaker message: m vectors (or text for kind == 'text')."""
        self.reset()
        prompt = (U + f"You are Bob, the detective. Last night you learned that {secret} is the mafioso. "
                  "Send Charlie a short message that tells him who the mafioso is." + A)
        logits, h = self.feed_prompt_with_hidden(prompt)
        if kind == "text":
            out = []
            for _ in range(steps):
                t = int(np.argmax(logits))
                if t == self.llm.token_eos():
                    break
                out.append(t)
                self.decode(tokens=[t])
                logits = self.logits()
            return self.llm.detokenize(out).decode("utf-8", "ignore")
        vecs = []
        for _ in range(steps):
            e = self.translate(kind, h, logits, rng)
            vecs.append(e)
            self.decode(embs=e[None, :], hidden=True)
            logits, h = self.logits(), self.hidden()
        return np.array(vecs, dtype=np.float32)

    def listen(self, message, candidates):
        self.reset()
        head = U + "You are Charlie, a villager in a game of Mafia. Bob sent you this message: "
        tail = (NL + "Who does Bob say is the mafioso? Answer with one name from: " + ", ".join(candidates) + "." + A)
        if message is None:
            self.decode(tokens=self.tok(U + "You are Charlie, a villager in a game of Mafia. Bob sent you no message."
                                        + tail))
        elif isinstance(message, str):
            self.decode(tokens=self.tok(head + message + tail))
        else:
            self.decode(tokens=self.tok(head))
            self.decode(embs=message)
            self.decode(tokens=self.tok(tail))
        logits = self.logits()
        # score each candidate by the probability of its first token as the answer's first token
        firsts = [self.tok(c)[0] for c in candidates]
        return candidates[int(np.argmax(logits[firsts]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    eng = Engine()
    print(f"loaded in {time.time() - t0:.0f}s; mean input-embedding norm {eng.mean_norm:.3f}")
    rng = np.random.default_rng(0)
    kinds = ["none", "text", "hard", "soft-0.5", "soft-1.0", "identity", "noise"]
    res = {}
    for kind in kinds:
        right = 0
        for i in range(args.trials):
            secret = NAMES[i % 3]
            cands = list(NAMES)
            rng.shuffle(cands)
            msg = None if kind == "none" else eng.speak(secret, kind, args.steps, rng)
            right += eng.listen(msg, cands) == secret
        res[kind] = right / args.trials
        extra = ""
        if kind not in ("none", "text"):
            v = eng.speak(NAMES[2], kind, args.steps, rng)
            En = eng.E / np.linalg.norm(eng.E, axis=1, keepdims=True)
            cos = [(En @ (x / np.linalg.norm(x))).max() for x in v[:8]]
            extra = f" | cos to nearest real token embedding {np.mean(cos):.2f}"
        elif kind == "text":
            extra = f" | e.g. {eng.speak(NAMES[2], 'text', args.steps, rng)!r}"
        print(f"{kind:9} listener names the right person {right}/{args.trials} = {right / args.trials:.0%}{extra}")
    (HERE / "logs" / "translation_test.json").write_text(json.dumps({"steps": args.steps, "trials": args.trials,
                                                                     "accuracy": res}, indent=1))


if __name__ == "__main__":
    main()
