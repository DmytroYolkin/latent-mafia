# latent-mafia

Can LLM agents play Mafia better when they talk through latent channels (hidden states / KV cache) instead of text?

## Contents

- `mafia_table/`: a Flask app that runs Mini-Mafia games (1 mafioso, 1 detective, 2 villagers) with a local Qwen3-4B GGUF model and shows them live at a table in the browser. It supports a text channel (T) and latent channels (L-16, L-thought, L-full, H, L-noise). Every game event is appended to `games.jsonl`; `stats.py` and `latent_insights.py` summarise the results.

## Running mafia_table

The current version runs the game engine from [bastoscostadavi/llm-mafia-game](https://github.com/bastoscostadavi/llm-mafia-game), cloned next to `mafia_table/` (not included here):

```
Project/
  llm-mafia-game/      # git clone of the benchmark repo, with models/Qwen3-4B-Q4_K_M.gguf
  mafia_table/
```

```
llm-mafia-game\.venv\Scripts\python mafia_table\app.py          # real model
llm-mafia-game\.venv\Scripts\python mafia_table\app.py --mock   # scripted, no model
```

Then open http://127.0.0.1:5055.
