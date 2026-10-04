"""Model backends for the game table.

LlamaBackend: one llama.cpp instance of Qwen3-4B (GGUF), shared by all four players.
MockBackend:  scripted lines, no model; used to build and test the interface.

Both expose generate(prompt, max_tokens, agent, on_token, thinking, think_budget) -> dict, called by
the repo's MafiaAgent through PlayerLLM (see bridge.py). on_token(delta, phase) streams text, where
phase is "think" (private thoughts, thinking mode only) or "answer" (what the repo's parser reads).
"""
import glob
import os
import random
import re
import site
import threading
import time

# Qwen3 chat template rendered with enable_thinking=False. The empty think block makes the
# model answer directly; LlamaBackend checks this against the template stored in the GGUF.
CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
# Thinking mode: the template's default generation prompt; the model opens <think> itself.
CHAT_SUFFIX_THINK = "<|im_end|>\n<|im_start|>assistant\n"
THINK_CLOSE = "\n</think>\n\n"  # appended when a thought runs past its budget
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S)

# Sampling presets, chosen per game; keyed by thinking on/off.
SAMPLING_PRESETS = {
    # The repo's Local wrapper for GGUF models (src/agent_interfaces.py), with the temperature recorded
    # for every local model in the benchmark database (players.temperature).
    "repo": {False: dict(temp=0.3, top_p=0.9, top_k=40, min_p=0.05, repeat_penalty=1.1),
             True: dict(temp=0.3, top_p=0.9, top_k=40, min_p=0.05, repeat_penalty=1.1)},
    # Qwen3-4B-GGUF README: thinking 0.6 / 0.95 / 20 / 0, non-thinking 0.7 / 0.8 / 20 / 0, and
    # presence_penalty 1.5 "for quantized models to suppress repetitive outputs"; no repeat penalty.
    "qwen": {True: dict(temp=0.6, top_p=0.95, top_k=20, min_p=0.0, repeat_penalty=1.0, presence_penalty=1.5),
             False: dict(temp=0.7, top_p=0.8, top_k=20, min_p=0.0, repeat_penalty=1.0, presence_penalty=1.5)},
}
SAMPLING = SAMPLING_PRESETS["repo"][False]


class ContextOverflow(RuntimeError):
    pass


def add_cuda_dll_dirs():
    """The cu125 llama-cpp-python wheel links cudart/cublas; they come from the nvidia-* pip wheels."""
    for sp in site.getsitepackages():
        dirs = glob.glob(os.path.join(sp, "nvidia", "*", "bin"))
        for d in dirs:
            os.add_dll_directory(d)
        if dirs:
            os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])


class LlamaBackend:
    mock = False

    def __init__(self, model_path, n_ctx=4096):
        add_cuda_dll_dirs()
        from llama_cpp import Llama

        self.model_path = str(model_path)
        self.name = os.path.basename(self.model_path)
        self.size_bytes = os.path.getsize(self.model_path)
        self.n_ctx = n_ctx
        self.temperature = SAMPLING["temp"]
        self.llm = Llama(model_path=self.model_path, n_gpu_layers=-1, n_ctx=n_ctx, verbose=False)
        self.n_layers = self.llm.metadata.get("qwen3.block_count")
        self.lock = threading.Lock()
        self.max_prompt_tokens = 0
        self.think_stripped = 0
        self.template_check = self._check_template()
        # End-of-turn tokens: <|im_end|> (the GGUF's EOS) and <|endoftext|>.
        self.stop_ids = {self.llm.token_eos()}
        for t in (b"<|im_end|>", b"<|endoftext|>"):
            self.stop_ids.update(self.llm.tokenize(t, add_bos=False, special=True))
        (self.think_close_id,) = self.llm.tokenize(b"</think>", add_bos=False, special=True)
        self.close_ids = self.llm.tokenize(THINK_CLOSE.encode(), add_bos=False, special=True)

    def _check_template(self):
        tpl = self.llm.metadata.get("tokenizer.chat_template")
        if not tpl:
            return "no chat template in GGUF"
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(msg):
            raise ValueError(msg)

        env.globals["raise_exception"] = raise_exception
        msgs = [{"role": "user", "content": "PROMPT"}]
        off = env.from_string(tpl).render(messages=msgs, add_generation_prompt=True, enable_thinking=False)
        on = env.from_string(tpl).render(messages=msgs, add_generation_prompt=True, enable_thinking=True)
        if off != CHAT_PREFIX + "PROMPT" + CHAT_SUFFIX:
            return f"MISMATCH (thinking off): {off!r}"
        if on != CHAT_PREFIX + "PROMPT" + CHAT_SUFFIX_THINK:
            return f"MISMATCH (thinking on): {on!r}"
        return "matches GGUF template (enable_thinking False and True)"

    def set_seed(self, seed):
        self.llm.set_seed(seed)

    def _decode(self, toks):
        return self.llm.detokenize(toks).decode("utf-8", errors="ignore")

    def generate(self, prompt, max_tokens, agent=None, on_token=None, thinking=False, think_budget=1024,
                 stop_fn=None, force=None, sampling="repo"):
        """max_tokens is the answer limit (repo: 55 speech / 5 vote). In thinking mode the thought gets
        its own budget first; if it runs out we close it with </think> and the model answers.
        force = {"marker": regex, "inject": text, "budget": n}: if the answer lacks the marker (e.g. the
        SAY:/VOTE: line), append `inject` and generate up to n more tokens.
        sampling: a key of SAMPLING_PRESETS."""
        params = SAMPLING_PRESETS[sampling][thinking]
        suffix = CHAT_SUFFIX_THINK if thinking else CHAT_SUFFIX
        ids = self.llm.tokenize((CHAT_PREFIX + prompt + suffix).encode("utf-8"), add_bos=True, special=True)
        need = (len(ids) + max_tokens + (think_budget + len(self.close_ids) + 4 if thinking else 0)
                + (force["budget"] + 4 if force else 0))
        if need > self.n_ctx:
            raise ContextOverflow(f"prompt is {len(ids)} tokens, needs {need} with generation > n_ctx {self.n_ctx}")
        self.max_prompt_tokens = max(self.max_prompt_tokens, len(ids))
        think, answer, forced = [], [], False
        with self.lock:
            t0 = time.perf_counter()
            ctx = ids
            if thinking:
                shown, closed = "", False
                for tok in self.llm.generate(ids, **params):
                    if tok in self.stop_ids:
                        break
                    think.append(tok)
                    if tok == self.think_close_id:
                        closed = True
                        break
                    text = self._decode(think).replace("<think>", "").lstrip()
                    if on_token and len(text) > len(shown):
                        on_token(text[len(shown):], "think")
                        shown = text
                    if len(think) >= think_budget:
                        break
                if not closed:
                    forced = True
                    think += self.close_ids
                ctx = ids + think
            shown, lead = "", 0

            def run(context, budget):
                """Generate into `answer` until end of turn, stop_fn, or `budget` new tokens."""
                nonlocal shown, lead
                start = len(answer)
                for tok in self.llm.generate(context, **params):
                    if tok in self.stop_ids:
                        return
                    answer.append(tok)
                    text = self._decode(answer)
                    if not text.strip():
                        lead += 1  # the newlines after </think> do not count against the answer limit
                    if on_token and len(text) > len(shown):
                        on_token(text[len(shown):], "answer")
                        shown = text
                    if len(answer) - start - lead >= budget or (stop_fn and stop_fn(text)):
                        return

            run(ctx, max_tokens)
            line_forced = False
            m = force and force["marker"].search(self._decode(answer).replace("\\n", "\n"))
            if force and (not m or not m.group(1).strip()):
                # Private notes ran out (or the model stopped) before the SAY:/VOTE: line, or the line
                # is empty: close the notes with that label and let the model write the line, like
                # closing a long thought.
                line_forced = True
                inject = force["inject"] if not m else " "
                answer += self.llm.tokenize(inject.encode("utf-8"), add_bos=False, special=False)
                lead = 0
                run(ctx + answer, force["budget"])
            seconds = time.perf_counter() - t0
        text = self.llm.detokenize(answer).decode("utf-8", errors="replace")
        stripped = "<think>" in text
        if stripped:  # guard: the parser must never see a think block
            text = THINK_RE.sub("", text)
            self.think_stripped += 1
        thought = self.llm.detokenize(think).decode("utf-8", errors="replace")
        thought = thought.replace("<think>", "").replace("</think>", "").strip()
        # The repo's Local wrapper strips whitespace before parsing; keep that.
        return {"text": text.strip(), "tokens": len(think) + len(answer), "think_tokens": len(think),
                "thinking": thought if thinking else None, "think_forced": forced, "line_forced": line_forced,
                "prompt_tokens": len(ids), "seconds": seconds, "think_stripped": stripped}


class MockBackend:
    """Scripted players: no model, realistic timing, and some malformed outputs on purpose."""

    mock = True
    name = "mock (scripted lines, no model)"
    size_bytes = 0
    n_ctx = 4096
    n_layers = None
    temperature = None
    template_check = "n/a (mock)"
    max_prompt_tokens = 0
    think_stripped = 0

    def __init__(self, delay=0.035):
        self.delay = delay
        self.rng = random.Random()

    def set_seed(self, seed):
        self.rng.seed(seed)

    def _context(self, agent, prompt):
        names = re.search(r"other players are: (.+)\.", prompt).group(1).split(", ")
        mem = agent.get_memory()
        victim = re.search(r"(\w+) was found dead", mem).group(1)
        alive = [n for n in names if n != victim]
        maf = re.search(r"You investigated (\w+)", mem)
        return alive, victim, maf.group(1) if maf else None

    def _speech(self, agent, prompt):
        alive, victim, maf = self._context(agent, prompt)
        a, b = alive[0], alive[-1]
        r = self.rng.random()
        if r < 0.10:
            return f"I think we should wait and see what {a} says."  # no quotes -> repo parser fails
        if agent.role == "detective":
            lines = [f'"I am the detective. I investigated {maf} last night: {maf} is the mafioso. Vote {maf}!"',
                     f'"{maf} is the mafioso, I checked them. {victim} is gone, do not let {maf} fool you."']
        elif agent.role == "mafioso":
            lines = [f'"I am the detective! I investigated {a} and {a} is the mafioso."',
                     f'"{b} has been very quiet since {victim} died. That worries me a lot."',
                     f'"Why would {a} accuse me so fast? That is exactly what the mafia would do."']
        else:
            lines = [f'"{a}, {b}: you cannot both be right. Who can prove what they saw last night?"',
                     f'"{victim} is dead and one of you two did it. {a}, explain yourself before we vote."']
        return self.rng.choice(lines) + "\n I am saying this because it fits what I know."

    def _vote(self, agent, prompt):
        cands = re.search(r"Vote to arrest one player from: (.+)\.", prompt).group(1).split(", ")
        if self.rng.random() < 0.08:
            return "Hard to say, everyone sounds honest."  # no name -> repo falls back to a random vote
        _, _, maf = self._context(agent, prompt)
        if agent.role == "detective" and maf in cands:
            return maf
        return self.rng.choice(cands)

    def generate(self, prompt, max_tokens, agent=None, on_token=None, thinking=False, think_budget=1024,
                 stop_fn=None, force=None, sampling="repo"):
        t0 = time.perf_counter()
        thought = []
        if thinking:
            thought = re.findall(r"\S+\s*", f"Mock thoughts of {agent.name}: I am the {agent.role}. "
                                            "Let me look at my memory and decide what to say. " * 2)
            for w in thought:
                time.sleep(self.delay / 2)
                if on_token:
                    on_token(w, "think")
        text = self._vote(agent, prompt) if "#VOTING TIME" in prompt else self._speech(agent, prompt)
        if "#VOTING TIME" in prompt and "VOTE:" in prompt[prompt.rfind("#VOTING TIME"):]:  # VOTE: line protocols
            text = f"PRIVATE: I am the {agent.role}.\nVOTE: {text}\n"
        if "#VOTING TIME" not in prompt and "SAY:" in prompt:  # private-say protocol
            msg = text.split("\n")[0]
            text = (f"PRIVATE: I am the {agent.role}, so I must not give that away.\nSAY: {msg.strip(chr(34))}\n"
                    if msg.startswith('"') else f"PRIVATE: {msg}")  # unquoted line -> no SAY: -> silent
        words = re.findall(r"\S+\s*", text)[: max(1, max_tokens)]
        for w in words:
            time.sleep(self.delay)
            if on_token:
                on_token(w, "answer")
        text = "".join(words)
        return {"text": text.strip(), "tokens": len(words) + len(thought), "think_tokens": len(thought),
                "thinking": "".join(thought).strip() if thinking else None, "think_forced": False,
                "line_forced": False,
                "prompt_tokens": len(prompt) // 4, "seconds": time.perf_counter() - t0, "think_stripped": False}
