"""The "own words" speech protocol.

The repo asks for '"public message" <newline> private reasoning' in one reply and keeps only the
quoted first line. Small models break this: they put the reasoning inside the quotes, cut the
message mid-sentence, or copy the previous speaker. Here a discussion reply is only what the player
says out loud (private reasoning, if any, lives in the think block), the parser below extracts it
defensively, and a reply that repeats an earlier message is regenerated.

Installed into the repo's agent module per game by bridge.py; the repo's files are not edited.
"""
import difflib
import re
from pathlib import Path

from src import prompt_utils

MESSAGE_LIMIT = 200
NAMES = ("Alice", "Bob", "Charlie", "Diana", "You")
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S)
LABEL_RE = re.compile(r"^\s*[*_]*(?:public\s+message|my\s+message|message|response|reply|"
                      + "|".join(NAMES) + r")[*_]*\s*:\s*", re.I)
# Where private reasoning starts, if a model writes some anyway.
REASONING_RE = re.compile(r"\\n|\n|\(?\b(?:reasoning|reason|explanation|private|note)\s*:|"
                          r"\bI(?:'m| am) saying this because\b", re.I)
MEMORY_MSG_RE = re.compile(r'^(?:\w+): "(.*)"$')
COPY_RATIO = 0.85


def format_discussion_prompt(name, other_players, composition, memory, round_num, discussion_rounds):
    """Same signature as the repo's builder; only the closing instruction differs."""
    suffix = (f"#DISCUSSION ROUND {round_num}/{discussion_rounds}:\n"
              "What do you say to everyone? Reply with your message only:\n")
    return prompt_utils._BASE_PROMPT.format(name=name, other_players=other_players, composition=composition,
                                            memory=memory, action_specific_content=suffix)


def parse_discussion_response(response, limit=MESSAGE_LIMIT):
    """Public message from a reply, or None (the engine then says the player remained silent)."""
    text = THINK_RE.sub("", response or "").strip()
    while True:  # drop labels such as 'Diana:' or 'Message:' (possibly on their own line)
        stripped = LABEL_RE.sub("", text, count=1).lstrip()
        if stripped == text:
            break
        text = stripped
    quoted = re.match(r'^["“]([^"”\n]+)["”]', text)
    if quoted:
        text = quoted.group(1)
    text = REASONING_RE.split(text, maxsplit=1)[0].lstrip('"“')
    text = " ".join(text.split()).strip(' "“”')
    if limit and len(text) > limit:
        cut = text[:limit]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[:end + 1] if end >= 60 else cut[:cut.rfind(" ")] + "…"
    return text or None


def parse_voting_response(response, candidates):
    """Exact name on the first line, else the candidate mentioned first (not first in list order)."""
    text = THINK_RE.sub("", response or "").replace("\\n", "\n").strip()
    first = text.split("\n")[0].strip().strip('."*:!“”"').strip()
    for c in candidates:
        if c.lower() == first.lower():
            return c
    hits = [(m.start(), c) for c in candidates if (m := re.search(rf"\b{re.escape(c)}\b", text, re.I))]
    return min(hits)[1] if hits else None


def _norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def earlier_messages(memory_lines):
    return [m.group(1) for line in memory_lines if (m := MEMORY_MSG_RE.match(line))]


def copied_from(message, earlier):
    """The earlier message this one repeats (exactly or nearly), or None."""
    n = _norm(message)
    for e in earlier:
        if n == _norm(e) or difflib.SequenceMatcher(None, n, _norm(e)).ratio() >= COPY_RATIO:
            return e
    return None


def retry_prompt(prompt, copy, closing="Reply with your message only:"):
    return prompt + (f'\nYour reply "{copy}" repeats a message already said in this game. '
                     f"Say something new, in your own words. {closing}\n")


# ---- "private-say" protocol: PRIVATE: notes first, then the SAY: line -----------------------
# Speech-only replies leak: with nowhere private to reason, a small model says "I'm the mafioso,
# I killed Bob" out loud. Here the reply has a private part, and only the SAY: line is public.
# No SAY: line means silence, so private notes can never be published by accident.
NOTES_BUDGET = 120  # tokens for the private notes, on top of the repo's 55 for the message
PRIVATE_SAY_CLOSING = "Write PRIVATE: then SAY:"
SAY_RE = re.compile(r"^[\s*_#>-]*SAY[*_]*\s*:[*_]*[ \t]*(.*)$", re.I | re.M)
PRIVATE_RE = re.compile(r"^[\s*_#>-]*PRIVATE[*_]*\s*:[*_]*\s*", re.I)


def format_private_say_prompt(name, other_players, composition, memory, round_num, discussion_rounds):
    suffix = (f"#DISCUSSION ROUND {round_num}/{discussion_rounds}:\n"
              f"What do you say to everyone? {PRIVATE_SAY_CLOSING}\n")
    return prompt_utils._BASE_PROMPT.format(name=name, other_players=other_players, composition=composition,
                                            memory=memory, action_specific_content=suffix)


def split_reply(response):
    """(private notes, raw SAY text or None)."""
    text = THINK_RE.sub("", response or "").replace("\\n", "\n").strip()
    m = SAY_RE.search(text)
    if not m:
        return PRIVATE_RE.sub("", text).strip(), None
    notes = PRIVATE_RE.sub("", text[:m.start()]).strip()
    say = m.group(1).strip() or next((l.strip() for l in text[m.end():].split("\n") if l.strip()), "")
    return notes, say


def parse_private_say(response):
    _, say = split_reply(response)
    return parse_discussion_response(say) if say else None


def _line_done(regex, text):
    text = text.replace("\\n", "\n")
    m = regex.search(text)
    if not m:
        return False
    rest = text[m.start():].split(":", 1)[1].lstrip()
    return bool(rest) and "\n" in rest


def say_done(text):
    """Stop generating once the SAY: line has content and has ended."""
    return _line_done(SAY_RE, text)


# Votes in the same shape: PRIVATE: notes, then VOTE: name. No VOTE: line means a random vote.
VOTE_RE = re.compile(r"^[\s*_#>-]*VOTE[*_]*\s*:[*_]*[ \t]*(.*)$", re.I | re.M)
PRIVATE_VOTE_CLOSING = "Write PRIVATE: then VOTE:"


def format_private_vote_prompt(name, other_players, composition, memory, candidates):
    suffix = f"#VOTING TIME: \nVote to arrest one player from: {candidates}.\n{PRIVATE_VOTE_CLOSING}\n"
    return prompt_utils._BASE_PROMPT.format(name=name, other_players=other_players, composition=composition,
                                            memory=memory, action_specific_content=suffix)


def split_vote(response):
    """(private notes, raw VOTE text or None)."""
    text = THINK_RE.sub("", response or "").replace("\\n", "\n").strip()
    m = VOTE_RE.search(text)
    if not m:
        return PRIVATE_RE.sub("", text).strip(), None
    vote = m.group(1).strip() or next((l.strip() for l in text[m.end():].split("\n") if l.strip()), "")
    return PRIVATE_RE.sub("", text[:m.start()]).strip(), vote


def parse_private_vote(response, candidates):
    _, vote = split_vote(response)
    return parse_voting_response(vote, candidates) if vote else None


def vote_done(text):
    """Stop generating once the VOTE: line has content and has ended."""
    return _line_done(VOTE_RE, text)


# ---- "briefed" protocol: role briefing + private notes (or thinking) + SAY:/VOTE: ------------
# Same reply shape as private-say, plus: a personal role section at the top of the prompt, the
# one-night/one-day/one-vote structure spelled out, round-aware closing lines, guidance on how to
# speak (no length rule), and no 200-character cap on the spoken message. With thinking on, the
# think block is the private space and the reply is just the SAY:/VOTE: line.
BRIEFED_DIR = Path(__file__).resolve().parent / "prompts" / "briefed"
SAY_BUDGET = 160    # tokens for the spoken message (not mentioned to the model)
VOTE_BUDGET = 12    # tokens for the "VOTE: name" line
ROLE_RE = re.compile(r"You're (\w+), the (\w+)\.")
# Set per game by bridge.py. "roles" picks the role briefings: prompts/briefed/ (first version) or
# prompts/fair/ (same structure and the same kind of facts for every role).
BRIEFED = {"thinking": False, "roles": "briefed"}


def _read(name, folder=None):
    return ((BRIEFED_DIR.parent / folder if folder else BRIEFED_DIR) / name).read_text(encoding="utf-8").strip()


def briefed_template(thinking):
    return _read("base.txt").replace("[[FORMAT]]", _read("format_think.txt" if thinking else "format_notes.txt"))


def _role_briefing(name, memory):
    m = ROLE_RE.search(memory)
    role = m.group(2) if m else "villager"
    return _read(f"role_{role}.txt", BRIEFED["roles"]).replace("{name}", name)


def briefed_closing(kind):
    if kind == "speech":
        return "Write SAY:" if BRIEFED["thinking"] else "Write PRIVATE: then SAY:"
    return "Write VOTE:" if BRIEFED["thinking"] else "Write PRIVATE: then VOTE:"


def format_briefed_discussion(name, other_players, composition, memory, round_num, discussion_rounds):
    left = ("After this round there is one more round, then the vote." if round_num < discussion_rounds
            else "This is the last round. Right after it, everyone votes and the game ends.")
    suffix = (f"#NOW: DISCUSSION ROUND {round_num} OF {discussion_rounds}\n{left}\n"
              f"It is your turn to speak. {briefed_closing('speech')}\n")
    return prompt_utils._BASE_PROMPT.format(name=name, other_players=other_players, composition=composition,
                                            memory=memory, action_specific_content=suffix,
                                            role_briefing=_role_briefing(name, memory))


def format_briefed_vote(name, other_players, composition, memory, candidates):
    suffix = (f"#VOTING TIME: THE FINAL VOTE (the game ends right after it)\n"
              f"Vote to arrest one player from: {candidates}.\n{briefed_closing('vote')}\n")
    return prompt_utils._BASE_PROMPT.format(name=name, other_players=other_players, composition=composition,
                                            memory=memory, action_specific_content=suffix,
                                            role_briefing=_role_briefing(name, memory))


def parse_briefed_say(response):
    """The SAY: line, with no length cap; a message cut off mid-sentence is trimmed to its last
    complete sentence."""
    _, say = split_reply(response)
    text = parse_discussion_response(say, limit=None) if say else None
    if text and not text.endswith((".", "!", "?", '"', "”", ")", "…", "⟧")):  # ⟧ ends a latent marker
        end = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
        text = text[:end + 1] if end >= len(text) * 0.3 else text + "…"
    return text
