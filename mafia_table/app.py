"""Mini-Mafia game table: plays the repo's Mini-Mafia with Qwen3-4B and streams it to a browser.

    llm-mafia-game\\.venv\\Scripts\\python mafia_table\\app.py          # real model
    llm-mafia-game\\.venv\\Scripts\\python mafia_table\\app.py --mock   # scripted, no model

Then open http://127.0.0.1:5055
"""
import argparse
import json
import queue
import random
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bridge  # noqa: E402
from gpu import GpuWatch  # noqa: E402
from latent import CHANNELS  # noqa: E402
from llm import SAMPLING_PRESETS, ContextOverflow, LlamaBackend, MockBackend  # noqa: E402


class Hub:
    """Fan-out of server-sent events to every open browser tab."""

    def __init__(self):
        self.subs = set()
        self.lock = threading.Lock()

    def subscribe(self):
        q = queue.Queue(maxsize=5000)
        with self.lock:
            self.subs.add(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.subs.discard(q)

    def publish(self, msg):
        with self.lock:
            for q in list(self.subs):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    self.subs.discard(q)


class EventLog:
    """games.jsonl: one JSON event per line, appended as the game is played."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def append(self, ev):
        with self.lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def read(self):
        if not self.path.exists():
            return []
        with self.lock, self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


class Table:
    def __init__(self, args):
        self.args = args
        self.hub = Hub()
        self.log = EventLog(args.log)
        self.jobs = queue.Queue()    # batches
        self.games = queue.Queue()   # single games; these jump ahead of batches
        self.gpu = GpuWatch(budget_gib=args.budget_gib)
        self.backend = None
        self.stop_requested = False  # stops the running batch (single games are not affected)
        self.current = []        # events of the game being played (late-joining tabs catch up)
        self.status = {"state": "starting", "error": None, "running": None,
                       "queued": {"games": 0, "batches": 0}}
        (HERE / "logs").mkdir(exist_ok=True)

    # ---- worker -------------------------------------------------------------------------
    def load(self):
        self.gpu.start()
        self.status["state"] = "loading model"
        try:
            if self.args.mock:
                self.backend = MockBackend()
            else:
                if not bridge.MODEL_PATH.exists():
                    raise FileNotFoundError(f"model not found: {bridge.MODEL_PATH}")
                self.backend = LlamaBackend(bridge.MODEL_PATH, n_ctx=self.args.n_ctx)
                time.sleep(2.5)  # let the shared-memory sampler see the loaded state
                self.gpu.mark_loaded()
            self.status["state"] = "ready"
        except Exception as e:
            self.fail(f"model load failed: {e}")

    def fail(self, msg):
        self.status["state"] = "stopped"
        self.status["error"] = msg
        self.stop_requested = True
        print("STOPPED:", msg, file=sys.stderr)
        self.push_status()

    def emit(self, ev):
        self.current.append(ev)
        self.log.append(ev)
        self.hub.publish({"kind": "event", "event": ev})

    def live(self, kind, **fields):
        self.hub.publish({"kind": kind, **fields})

    def run_game(self, config, batch_id=None, index=None, size=None):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        game_id = f"{stamp}-{random.randrange(16**4):04x}" + ("-mock" if self.args.mock else "")
        seed = random.randrange(2**31)
        self.current = []
        self.status["running"] = {"game_id": game_id, "batch_id": batch_id, "index": index, "size": size,
                                  "config": config}
        self.push_status()
        try:
            winner, engine_log = bridge.play_game(self.backend, self.emit, self.live, game_id, seed,
                                                  batch_id=batch_id, game_index=index, batch_size=size,
                                                  config=config)
            (HERE / "logs" / f"{game_id}.txt").write_text(engine_log, encoding="utf-8")
        except ContextOverflow as e:
            self.fail(f"prompt truncated in {game_id}: {e}. Raise --n-ctx.")
        except Exception as e:
            text = str(e)
            oom = any(s in text.lower() for s in ("out of memory", "cudamalloc", "failed to allocate", "llama_decode"))
            traceback.print_exc()
            self.fail(("out of memory" if oom else "error") + f" in {game_id}: {text}")
        finally:
            self.status["running"] = None
        if self.gpu.problem:
            self.fail(f"VRAM limit hit during {game_id}: {self.gpu.problem}")

    def queue_status(self):
        self.status["queued"] = {"games": self.games.qsize(), "batches": self.jobs.qsize()}

    def run_single_games(self):
        """Single games ("New game") jump ahead of batches: they run between batch games."""
        while not self.status["error"]:
            try:
                job = self.games.get_nowait()
            except queue.Empty:
                return
            self.queue_status()
            self.run_game(job["config"])

    def worker(self):
        self.load()
        while True:
            if not self.games.empty():
                self.status["state"] = "playing"
                self.run_single_games()
                self.idle()
            try:
                job = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            self.queue_status()
            if self.status["error"]:
                continue  # stopped on error: refuse further work until restart
            self.stop_requested = False
            self.status["state"] = "playing"
            config = job["config"]
            label = (f"{config['channel']}-{config['prompt']}-"
                     f"{'think' + str(config['think_budget']) if config['thinking'] else 'nothink'}-{config['sampling']}"
                     + (f"-m{config['latent_steps']}" if config["channel"] != "T" else "")
                     + (f"-w{config['latent_window']}" if config["channel"] in ("L-thought", "L-full") else "")
                     + ("-soft" if config["channel"] != "T" and config.get("latent_translation") == "soft" else ""))
            batch_id = (datetime.now().strftime("batch-%Y%m%d-%H%M%S") + f"-{label}"
                        + ("-mock" if self.args.mock else ""))
            for i in range(job["n"]):
                self.run_single_games()
                if self.stop_requested or self.status["error"]:
                    break
                self.run_game(config, batch_id, i + 1, job["n"])
            self.idle()

    def idle(self):
        if not self.status["error"]:
            self.status["state"] = "ready"
        self.queue_status()
        self.push_status()

    # ---- status -------------------------------------------------------------------------
    def status_payload(self):
        b = self.backend
        return {
            **self.status,
            "mock": self.args.mock,
            "model": None if b is None else {
                "name": b.name, "size_bytes": b.size_bytes, "n_ctx": b.n_ctx, "n_layers": b.n_layers,
                "gpu_layers": "all" if not b.mock else None, "temperature": b.temperature,
                "thinking": "per game. off: empty <think></think> in the chat template (enable_thinking=False); "
                            "on: template default, thought capped at think_budget tokens, never shown to the parser",
                "template_check": b.template_check, "max_prompt_tokens": b.max_prompt_tokens,
                "think_stripped": b.think_stripped,
            },
            "gpu": self.gpu.snapshot(),
            "log": self.log.path.name,
            "prompts": list(bridge.PROMPTS),
            "sampling_presets": {k: {"thinking": v[True], "no_thinking": v[False]} for k, v in SAMPLING_PRESETS.items()},
            "default_config": self.default_config(),
            "channels": list(CHANNELS),
            "latent_engine": None if getattr(b, "_latent_engine", None) is None else {
                "n_ctx": b._latent_engine.n_ctx, "max_used": b._latent_engine.max_used,
                "embedding_norm": round(b._latent_engine.mean_norm, 4)},
        }

    def default_config(self):
        return {"prompt": self.args.prompt, "thinking": self.args.thinking, "think_budget": self.args.think_budget,
                "sampling": self.args.sampling, "channel": "T", "latent_steps": bridge.DEFAULT_CONFIG["latent_steps"],
                "latent_window": bridge.DEFAULT_CONFIG["latent_window"],
                "latent_translation": bridge.DEFAULT_CONFIG["latent_translation"]}

    def job_config(self, body):
        """Game settings from a request body, falling back to the command-line defaults."""
        cfg = self.default_config()
        if body.get("prompt") in bridge.PROMPTS:
            cfg["prompt"] = body["prompt"]
        if "thinking" in body:
            cfg["thinking"] = bool(body["thinking"])
        if body.get("think_budget"):
            cfg["think_budget"] = max(64, min(int(body["think_budget"]), 2048))
        if body.get("sampling") in SAMPLING_PRESETS:
            cfg["sampling"] = body["sampling"]
        if body.get("channel") in CHANNELS:
            cfg["channel"] = body["channel"]
        if body.get("latent_steps") is not None:
            cfg["latent_steps"] = max(0, min(int(body["latent_steps"]), 128))
        if body.get("latent_window") is not None:  # 0 = keep every turn (unmodified LatentMAS)
            cfg["latent_window"] = max(0, min(int(body["latent_window"]), 12))
        if body.get("latent_translation") in ("identity", "soft"):
            cfg["latent_translation"] = body["latent_translation"]
        if cfg["channel"] != "T":  # latent channels: no thinking block, PRIVATE:/SAY: reply format
            cfg["thinking"] = False
            if cfg["prompt"] not in bridge.BRIEFED_PROMPTS:
                cfg["prompt"] = "fair"
        return cfg

    def push_status(self):
        self.hub.publish({"kind": "status", "status": self.status_payload()})

    def status_loop(self):
        while True:
            time.sleep(1.0)
            if self.gpu.problem and self.status["running"] and not self.stop_requested:
                self.stop_requested = True  # stop the batch after this game; run_game reports it
            self.push_status()


def summarize(events):
    games = {}
    for ev in events:
        g = games.setdefault(ev["game_id"], {"game_id": ev["game_id"], "events": 0, "malformed": 0,
                                             "winner": None, "done": False})
        g["events"] += 1
        if ev["type"] == "start":
            g.update(ts=ev["ts"], batch_id=ev.get("batch_id"), game_index=ev.get("game_index"),
                     batch_size=ev.get("batch_size"), mock=ev.get("mock"), channel=ev["channel"],
                     prompt=ev.get("prompt", "repo"), thinking=bool(ev.get("thinking")))
        if ev.get("parsed_ok") is False:
            g["malformed"] += 1
        if ev["type"] == "result":
            g.update(winner=ev["winner"], done=True, arrested=ev.get("arrested"))
    return sorted(games.values(), key=lambda g: g.get("ts", 0), reverse=True)


def create_app(table):
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(HERE / "static", "index.html")

    @app.get("/api/status")
    def status():
        return jsonify(table.status_payload())

    @app.post("/api/game")
    def new_game():
        cfg = table.job_config(request.get_json(silent=True) or {})
        table.games.put({"kind": "game", "config": cfg})
        table.queue_status()
        return jsonify(ok=True, queued=table.status["queued"], config=cfg)

    @app.post("/api/batch")
    def batch():
        body = request.get_json(silent=True) or {}
        n = max(1, min(int(body.get("n", 30)), 500))
        cfg = table.job_config(body)
        table.jobs.put({"kind": "batch", "n": n, "config": cfg})
        table.queue_status()
        return jsonify(ok=True, n=n, config=cfg)

    @app.post("/api/stop")
    def stop():
        """Stop the running batch after its current game and drop queued batches. Single games stay."""
        while not table.jobs.empty():
            table.jobs.get_nowait()
        table.stop_requested = True
        table.queue_status()
        return jsonify(ok=True)

    @app.get("/api/games")
    def games():
        return jsonify(summarize(table.log.read()))

    @app.get("/api/events")
    def events():
        gid, bid = request.args.get("game_id"), request.args.get("batch_id")
        evs = table.log.read()
        if gid:
            evs = [e for e in evs if e["game_id"] == gid]
        elif bid:
            ids = {e["game_id"] for e in evs if e["type"] == "start" and e.get("batch_id") == bid}
            evs = [e for e in evs if e["game_id"] in ids]
        return jsonify(evs)

    @app.get("/games.jsonl")
    def raw_log():
        return send_from_directory(table.log.path.parent, table.log.path.name, mimetype="application/x-ndjson")

    @app.get("/api/stream")
    def stream():
        q = table.hub.subscribe()

        def gen():
            try:
                hello = {"kind": "hello", "status": table.status_payload(), "current": list(table.current)}
                yield f"data: {json.dumps(hello)}\n\n"
                while True:
                    try:
                        yield f"data: {json.dumps(q.get(timeout=15))}\n\n"
                    except queue.Empty:
                        yield ": ping\n\n"
            finally:
                table.hub.unsubscribe(q)

        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="scripted players, no model")
    ap.add_argument("--port", type=int, default=5055)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--budget-gib", type=float, default=6.5, help="stop if device VRAM use exceeds this")
    ap.add_argument("--prompt", choices=list(bridge.PROMPTS), default=bridge.DEFAULT_CONFIG["prompt"],
                    help="default prompt variant (the UI can override per game)")
    ap.add_argument("--thinking", action="store_true", help="default to Qwen3 thinking mode")
    ap.add_argument("--think-budget", type=int, default=bridge.DEFAULT_CONFIG["think_budget"],
                    help="max thinking tokens per reply")
    ap.add_argument("--sampling", choices=list(SAMPLING_PRESETS), default=bridge.DEFAULT_CONFIG["sampling"],
                    help="qwen: Qwen3 model-card settings; repo: the repo's local-model settings (temp 0.3)")
    ap.add_argument("--log", default=None, help="event log (default games.jsonl, or games.mock.jsonl with --mock)")
    args = ap.parse_args()
    args.log = Path(args.log) if args.log else HERE / ("games.mock.jsonl" if args.mock else "games.jsonl")

    table = Table(args)
    threading.Thread(target=table.worker, daemon=True).start()
    threading.Thread(target=table.status_loop, daemon=True).start()
    print(f"Mini-Mafia table on http://{args.host}:{args.port}  ({'mock' if args.mock else bridge.MODEL_FILE}, "
          f"log: {args.log})")
    create_app(table).run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
