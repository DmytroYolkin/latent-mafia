"""Runs one Mini-Mafia game with the repo's own engine and turns it into table events.

Nothing in the repo is modified. We call mini_mafia.create_mini_mafia_game() and game.play()
exactly as the benchmark does, and observe the game from outside:
  * each agent's .llm is swapped for PlayerLLM, which sends the engine's prompt to our backend
    (one shared model) and streams tokens to the browser;
  * each agent's message()/vote() is wrapped to emit one event after the engine has parsed the
    response (parse results come from the engine's own game_sequence log).
"""
import contextlib
import hashlib
import io
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "llm-mafia-game"
for p in (REPO, REPO / "mini-mafia-benchmark"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mini_mafia import create_mini_mafia_game  # noqa: E402  (repo engine)
from src import agent_interfaces, prompt_utils  # noqa: E402
from src import agents as repo_agents  # noqa: E402

import latent  # noqa: E402
import speech  # noqa: E402
from llm import SAMPLING_PRESETS  # noqa: E402

MODEL_FILE = "Qwen3-4B-Q4_K_M.gguf"
MODEL_PATH = REPO / "models" / MODEL_FILE
SEAT_ORDER = ["Alice", "Bob", "Charlie", "Diana"]  # names used by mini_mafia.py

# Prompt variants. "repo" is src/prompt.txt as shipped. "explained" (prompts/explained.txt) states the
# Mini-Mafia rules and goal precisely, with no strategy advice. Same placeholders, same response formats,
# so the repo's prompt builders and parsers work unchanged. Selected per game by swapping the template
# string that src/prompt_utils.py loaded at import; no repo file is modified.
PROMPTS = {
    "repo": prompt_utils.load_base_prompt(),
    "explained": (HERE / "prompts" / "explained.txt").read_text(encoding="utf-8").strip(),
    # Same rules as "explained", but a discussion reply is only the spoken message (see speech.py).
    # Kept for comparison: with no private space, the mafioso often says "I'm the mafioso" aloud.
    "own-words": (HERE / "prompts" / "own_words.txt").read_text(encoding="utf-8").strip(),
    # Same rules; reply = PRIVATE: notes, then SAY: message. Only the SAY: line is public.
    "private-say": (HERE / "prompts" / "private_say.txt").read_text(encoding="utf-8").strip(),
    # prompts/briefed/: personal role briefing, one-night/one-day/one-vote spelled out, guidance on
    # how to speak, longer messages. The template depends on thinking, so play_game rebuilds it.
    "briefed": speech.briefed_template(False),
    # Same as "briefed", with role briefings that give every role the same structure and the same
    # kind of facts (prompts/fair/); facts about claims are only in the shared rules.
    "fair": speech.briefed_template(False),
}
BRIEFED_PROMPTS = ("briefed", "fair")
DEFAULT_CONFIG = {"prompt": "fair", "thinking": False, "think_budget": 2048, "sampling": "qwen",
                  "channel": "T", "latent_steps": 16, "latent_window": 3, "latent_translation": "identity"}
MAX_COPY_RETRIES = 2

# The repo's agent module imports its prompt builder and parsers by name; our protocols swap those
# names for the game, the repo and explained prompts get the repo's originals back.
REPO_HOOKS = {n: getattr(repo_agents, n)
              for n in ("format_discussion_prompt", "parse_discussion_response",
                        "format_voting_prompt", "parse_voting_response")}
PROTOCOL_HOOKS = {
    "own-words": {**REPO_HOOKS,
                  "format_discussion_prompt": speech.format_discussion_prompt,
                  "parse_discussion_response": speech.parse_discussion_response,
                  "parse_voting_response": speech.parse_voting_response},
    "private-say": {"format_discussion_prompt": speech.format_private_say_prompt,
                    "parse_discussion_response": speech.parse_private_say,
                    "format_voting_prompt": speech.format_private_vote_prompt,
                    "parse_voting_response": speech.parse_private_vote},
    "briefed": {"format_discussion_prompt": speech.format_briefed_discussion,
                "parse_discussion_response": speech.parse_briefed_say,
                "format_voting_prompt": speech.format_briefed_vote,
                "parse_voting_response": speech.parse_private_vote},
}
PROTOCOL_HOOKS["fair"] = PROTOCOL_HOOKS["briefed"]


NOTES_PROTOCOLS = ("private-say", "briefed", "fair")  # replies carry PRIVATE: notes before SAY:/VOTE:


def retry_closing(protocol):
    return {"own-words": "Reply with your message only:", "private-say": speech.PRIVATE_SAY_CLOSING,
            "briefed": speech.briefed_closing("speech"), "fair": speech.briefed_closing("speech")}[protocol]


def install_protocol(prompt_name):
    for name, fn in PROTOCOL_HOOKS.get(prompt_name, REPO_HOOKS).items():
        setattr(repo_agents, name, fn)


class PlayerLLM:
    """Stands in for the repo's LLM wrapper: same generate(prompt, max_tokens) -> str contract."""

    def __init__(self, backend, agent, live, config, runner=None):
        self.backend, self.agent, self.live, self.config = backend, agent, live, config
        self.runner = runner  # latent channels: a latent.ChannelRunner shared by the game's players
        self.display_name = backend.name
        self.last = None

    def _force(self, kind):
        """With PRIVATE: notes, make sure the reply still ends with its SAY:/VOTE: line."""
        if self.config["prompt"] not in NOTES_PROTOCOLS:
            return None
        if kind == "vote":
            return {"marker": speech.VOTE_RE, "inject": "\nVOTE:", "budget": speech.VOTE_BUDGET}
        say_budget = speech.SAY_BUDGET if self.config["prompt"] in BRIEFED_PROMPTS else 55
        return {"marker": speech.SAY_RE, "inject": "\nSAY:", "budget": say_budget}

    def _call(self, prompt, max_tokens, kind, stop_fn=None):
        self.live("gen_start", speaker=self.agent.name, type=kind)
        return self.backend.generate(
            prompt, max_tokens, agent=self.agent, thinking=self.config["thinking"],
            think_budget=self.config["think_budget"], stop_fn=stop_fn, force=self._force(kind),
            sampling=self.config["sampling"],
            on_token=lambda delta, phase: self.live("token", speaker=self.agent.name, type=kind,
                                                    delta=delta, phase=phase))

    def generate(self, prompt, max_tokens=50):
        kind = "vote" if "#VOTING TIME" in prompt else "speech"
        stop_fn = None
        if self.config["prompt"] == "private-say":
            max_tokens += speech.NOTES_BUDGET  # private notes get their own budget
            stop_fn = speech.say_done if kind == "speech" else speech.vote_done
        elif self.config["prompt"] in BRIEFED_PROMPTS:
            # Own budgets instead of the repo's 55/5; notes only when the think block is not there.
            notes = 0 if self.config["thinking"] else speech.NOTES_BUDGET
            max_tokens = notes + (speech.SAY_BUDGET if kind == "speech" else speech.VOTE_BUDGET)
            stop_fn = speech.say_done if kind == "speech" else speech.vote_done
        if self.runner:
            # Latent channel: the runner decides what travels (vectors, shared cache, text + vectors).
            # A latent speaker's streamed text is its gloss (viewers only), so it is tagged as such.
            self.live("gen_start", speaker=self.agent.name, type=kind)
            gloss = kind == "speech" and self.config["channel"] != "H"
            r = self.runner.reply(
                self.agent, prompt, kind, max_tokens, SAMPLING_PRESETS[self.config["sampling"]][False],
                stop_fn, self._force(kind),
                on_token=lambda delta, phase: self.live("token", speaker=self.agent.name, type=kind, delta=delta,
                                                        phase="gloss" if gloss else phase))
            self.last = {**r, "retries": [], "copy_of": None, "think_tokens": 0, "thinking": None,
                         "think_forced": False, "think_stripped": False}
            return r["text"]
        protocol = self.config["prompt"] if kind == "speech" else None
        parse = PROTOCOL_HOOKS[protocol]["parse_discussion_response"] if protocol in PROTOCOL_HOOKS else None
        earlier = speech.earlier_messages(self.agent.memory) if parse else []
        calls, retries, ask, copy = [], [], prompt, None
        attempts = 1 + (MAX_COPY_RETRIES if earlier else 0)
        for attempt in range(attempts):
            calls.append(self._call(ask, max_tokens, kind, stop_fn))
            msg = parse(calls[-1]["text"]) if earlier else None
            copy = msg and speech.copied_from(msg, earlier)
            if not copy or attempt == attempts - 1:
                break  # accepted; if it is still a copy after the last retry, copy_of says so
            retries.append({"message": msg, "copy_of": copy})
            ask = speech.retry_prompt(prompt, copy, retry_closing(protocol))
        last = calls[-1]
        self.last = {**last, "retries": retries, "copy_of": copy or None,
                     "seconds": sum(c["seconds"] for c in calls), "tokens": sum(c["tokens"] for c in calls),
                     "think_tokens": sum(c["think_tokens"] for c in calls),
                     "think_forced": any(c["think_forced"] for c in calls),
                     "line_forced": calls[-1].get("line_forced", False),
                     "think_stripped": any(c["think_stripped"] for c in calls)}
        return last["text"]


def call_fields(call):
    """Per-reply measurements shared by speech and vote events."""
    return dict(seconds=call["seconds"], tokens=call["tokens"], prompt_tokens=call["prompt_tokens"],
                think_tokens=call["think_tokens"], thinking=call["thinking"], think_forced=call["think_forced"],
                think_stripped=call["think_stripped"], retries=call["retries"], copy_of=call["copy_of"],
                line_forced=call["line_forced"], latent=call.get("latent"))


def latent_engine(backend):
    """The latent llama.cpp context, created on first use (it reserves ~2 GB of KV cache)."""
    if getattr(backend, "_latent_engine", None) is None:
        backend._latent_engine = latent.LatentEngine(backend, cache_dir=HERE / "logs")
    return backend._latent_engine


def model_config(backend):
    return {"type": "local", "model": MODEL_FILE, "n_ctx": backend.n_ctx, "temperature": backend.temperature}


def play_game(backend, emit, live, game_id, seed, batch_id=None, game_index=None, batch_size=None, config=None):
    """Play one game. emit(event) persists an event; live(kind, **fields) sends transient updates."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    channel = config["channel"]
    if channel != "T":
        # Latent channels use the PRIVATE:/SAY: reply format (for votes, glosses and H's text) and no
        # thinking block: the latent steps are the thinking.
        config["thinking"] = False
        if config["prompt"] not in BRIEFED_PROMPTS:
            config["prompt"] = "fair"
    prompt_utils._BASE_PROMPT = PROMPTS[config["prompt"]]
    if config["prompt"] in BRIEFED_PROMPTS:
        speech.BRIEFED["roles"] = config["prompt"]
        speech.BRIEFED["thinking"] = config["thinking"]
        prompt_utils._BASE_PROMPT = speech.briefed_template(config["thinking"])
    install_protocol(config["prompt"])
    # Fingerprint of the exact prompt text used (template + role briefings), recorded in the start event.
    fingerprint = prompt_utils._BASE_PROMPT
    if config["prompt"] in BRIEFED_PROMPTS:
        fingerprint += "".join(speech._read(f"role_{r}.txt", config["prompt"]) for r in ("mafioso", "detective", "villager"))
    prompt_hash = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:10]
    random.seed(seed)
    backend.set_seed(seed)
    cfg = model_config(backend)
    runner = None
    if channel != "T":
        runner = latent.ChannelRunner(None if backend.mock else latent_engine(backend), channel,
                                      config["latent_steps"], config["sampling"], seed, game_id,
                                      HERE / "logs" / "latents", text_backend=backend if backend.mock else None,
                                      window=config["latent_window"] or None,
                                      translation=config["latent_translation"])

    # The repo loads GGUF models through a module-level cache keyed by path. Seed that cache with
    # our single shared instance so create_game() never loads a second copy of the model.
    root = os.path.dirname(os.path.dirname(os.path.abspath(agent_interfaces.__file__)))
    key = os.path.join(root, "models", MODEL_FILE)
    agent_interfaces._model_cache[key] = getattr(backend, "llm", None) or SimpleNamespace(model_path="mock")

    engine_log = io.StringIO()
    with contextlib.redirect_stdout(engine_log):
        game = create_mini_mafia_game({"detective": cfg, "mafioso": cfg, "villager": cfg})
    state = game.state
    agents = sorted(state.agents, key=lambda a: SEAT_ORDER.index(a.name) if a.name in SEAT_ORDER else 99)
    role_of = {a.name: a.role for a in agents}
    turn = [0]
    t_start = time.time()

    def event(type, phase, speaker=None, text="", target=None, winner=None, parsed_ok=None,
              seconds=0.0, tokens=0, **extra):
        ev = {"game_id": game_id, "turn": turn[0], "phase": phase, "channel": channel,
              "speaker": speaker, "role": role_of.get(speaker), "type": type, "text": text,
              "target": target, "winner": winner, "parsed_ok": parsed_ok,
              "seconds": round(seconds, 3), "tokens": tokens,
              "round": state.round or 1, "ts": round(time.time(), 3), **extra}
        turn[0] += 1
        emit(ev)

    def live_for_game(kind, **fields):
        live(kind, game_id=game_id, **fields)

    villagers = [a.name for a in agents if a.role == "villager"]
    event("start", "setup", text="Mini-Mafia: 1 mafioso, 1 detective, 2 villagers",
          players=[{"name": a.name, "role": a.role, "seat": i, "villager_no": villagers.index(a.name) if a.role == "villager" else None}
                   for i, a in enumerate(agents)],
          seed=seed, model=backend.name, mock=backend.mock, n_ctx=backend.n_ctx,
          prompt_hash=prompt_hash,
          sampling=config["sampling"], sampling_params=SAMPLING_PRESETS[config["sampling"]][config["thinking"]],
          temperature=SAMPLING_PRESETS[config["sampling"]][config["thinking"]]["temp"],
          discussion_rounds=state.discussion_rounds, batch_id=batch_id, game_index=game_index, batch_size=batch_size,
          prompt=config["prompt"], thinking=config["thinking"],
          think_budget=config["think_budget"] if config["thinking"] else None,
          latent_steps=config["latent_steps"] if channel != "T" else None,
          latent_window=config["latent_window"] if channel in latent.SHARED_CHANNELS else None,
          latent_translation=config["latent_translation"] if channel != "T" else None)

    # Night 1 is scripted by mini_mafia.py (no model calls). Read it back from the engine's log.
    for entry in state.game_sequence:
        actor, target = entry["actor"], entry["parsed_result"]
        if entry["action"] == "kill":
            event("night", "night", speaker=actor, target=target, text=f"{actor} killed {target}. {target} was found dead.",
                  scripted=True)
        elif entry["action"] == "investigate":
            event("night", "night", speaker=actor, target=target,
                  text=f"{actor} investigated {target} and learned they are the {role_of[target]}.", scripted=True)

    votes = {}
    for agent in agents:
        agent.llm = PlayerLLM(backend, agent, live_for_game, config, runner)
        orig_message, orig_vote = agent.message, agent.vote

        def message(active_players, round_num, all_players=None, discussion_rounds=2, game_state=None,
                    _a=agent, _f=orig_message):
            out = _f(active_players, round_num, all_players, discussion_rounds, game_state)
            entry = game_state.game_sequence[-1]
            ok = entry["parsed_result"] != "remained silent"
            call = _a.llm.last
            notes = speech.split_reply(entry["raw_response"])[0] if config["prompt"] in NOTES_PROTOCOLS else None
            event("speech", "day", speaker=_a.name, text=entry["parsed_result"] if ok else "remained silent.",
                  parsed_ok=ok, raw=entry["raw_response"], sub_round=round_num, notes=notes or None,
                  **call_fields(call))
            return out

        def vote(candidates, all_players=None, discussion_rounds=2, game_state=None, _a=agent, _f=orig_vote):
            target, ok = _f(candidates, all_players, discussion_rounds, game_state)
            entry = game_state.game_sequence[-1]
            call = _a.llm.last
            votes[_a.name] = target
            text = f"{_a.name} votes for {target}" + ("" if ok else " (random vote: could not parse)")
            notes = speech.split_vote(entry["raw_response"])[0] if config["prompt"] in NOTES_PROTOCOLS else None
            event("vote", "vote", speaker=_a.name, text=text, target=target, parsed_ok=ok,
                  raw=entry["raw_response"], notes=notes or None, **call_fields(call))
            return target, ok

        agent.message, agent.vote = message, vote

    with contextlib.redirect_stdout(engine_log):
        game.play()
    if runner:
        runner.save()

    # Same rule as experiments/run_mini_mafia_batch.py determine_winner().
    arrested = next((a for a in agents if a.imprisoned), None)
    winner = None if arrested is None else ("town" if arrested.role == "mafioso" else "mafia")
    counts = {}
    for t in votes.values():
        counts[t] = counts.get(t, 0) + 1
    top = max(counts.values()) if counts else 0
    tie = sum(1 for c in counts.values() if c == top) > 1
    mafioso = next(a.name for a in agents if a.role == "mafioso")
    if arrested:
        text = (f"{arrested.name} was arrested ({counts.get(arrested.name, 0)} of {len(votes)} votes"
                f"{', tie broken at random' if tie else ''}). {arrested.name} was the {arrested.role}. "
                f"{'Town' if winner == 'town' else 'Mafia'} wins.")
    else:
        text = "No one was arrested. Game incomplete."
    event("result", "end", target=arrested.name if arrested else None, winner=winner, text=text,
          arrested=arrested.name if arrested else None, mafioso=mafioso, votes=votes, tie=tie,
          roles=role_of, wall_seconds=round(time.time() - t_start, 2))
    return winner, engine_log.getvalue()
