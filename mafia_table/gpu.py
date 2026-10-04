"""VRAM watch.

Device memory comes from NVML. This laptop runs the GPU in WDDM mode with sysmem fallback on,
so running out of VRAM does not raise an error: allocations silently spill into shared system
RAM. We therefore also sample this process's "Shared Usage" GPU counter and treat growth
there as out-of-memory.
"""
import os
import subprocess
import threading
import time

GIB = 2**30
MIB = 2**20


class GpuWatch:
    def __init__(self, budget_gib=6.5, spill_mib=256, interval=0.5):
        self.budget = budget_gib * GIB
        self.spill_limit = spill_mib * MIB
        self.interval = interval
        self.ok = False
        self.used = self.peak = self.baseline = self.total = 0  # whole GPU (NVML), all processes
        self.proc = None            # this process's dedicated VRAM; the budget applies to this
        self.proc_peak = 0
        self.shared = None          # this process's shared GPU memory on the NVIDIA adapter
        self.shared_baseline = None
        self.problem = None         # set once: "over budget" or "spill"
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.name = pynvml.nvmlDeviceGetName(self._h)
            info = pynvml.nvmlDeviceGetMemoryInfo(self._h)
            self.total, self.used = info.total, info.used
            self.baseline = self.peak = self.used
            self.ok = True
        except Exception as e:  # no NVIDIA driver / NVML
            self.name = f"unavailable ({e})"

    def start(self):
        if self.ok:
            threading.Thread(target=self._poll_nvml, daemon=True).start()
            threading.Thread(target=self._poll_shared, daemon=True).start()

    def mark_loaded(self):
        """Call after the model is resident: later shared-memory growth means spilling."""
        self.shared_baseline = self.shared

    def _poll_nvml(self):
        while True:
            used = self._nvml.nvmlDeviceGetMemoryInfo(self._h).used
            self.used = used
            self.peak = max(self.peak, used)
            # Other programs share this GPU, so the budget is checked against this process's own
            # memory (sampled in _poll_shared). The whole-GPU number is only a fallback.
            if self.proc is None and used > self.budget and not self.problem:
                self.problem = f"over budget: {used / GIB:.2f} GiB used on the GPU > {self.budget / GIB:.1f} GiB"
            time.sleep(self.interval)

    def _poll_shared(self):
        pid = os.getpid()
        ps = (
            "while ($true) { try { "
            f"$s = (Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Shared Usage','\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage' -ErrorAction Stop).CounterSamples; "
            "$s | ForEach-Object { '{0}|{1}|{2}' -f $_.Path, $_.InstanceName, [int64]$_.CookedValue } ; 'END' "
            "} catch { 'END' } ; Start-Sleep -Seconds 2 }"
        )
        try:
            proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", ps], stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True, creationflags=0x08000000)
        except OSError:
            return
        rows = {}
        for line in proc.stdout:
            line = line.strip()
            if line != "END":
                parts = line.split("|")
                if len(parts) == 3:
                    path, inst, val = parts
                    kind = "shared" if "shared usage" in path.lower() else "dedicated"
                    rows.setdefault(inst, {})[kind] = int(val)
                continue
            if rows:
                # The NVIDIA adapter is the one where this process holds dedicated memory.
                nv = max(rows.values(), key=lambda r: r.get("dedicated", 0))
                self.shared = nv.get("shared", 0)
                self.proc = nv.get("dedicated", 0)
                self.proc_peak = max(self.proc_peak, self.proc)
                if self.proc > self.budget and not self.problem:
                    self.problem = (f"over budget: this app uses {self.proc / GIB:.2f} GiB of VRAM "
                                    f"> {self.budget / GIB:.1f} GiB")
                if (self.shared_baseline is not None and self.shared - self.shared_baseline > self.spill_limit
                        and not self.problem):
                    self.problem = (f"spill: {(self.shared - self.shared_baseline) / MIB:.0f} MiB moved to shared "
                                    f"system RAM (out of VRAM)")
            rows = {}

    def snapshot(self):
        return {
            "ok": self.ok, "device": self.name,
            "used_gib": round(self.used / GIB, 2), "peak_gib": round(self.peak / GIB, 2),
            "baseline_gib": round(self.baseline / GIB, 2), "total_gib": round(self.total / GIB, 2),
            "budget_gib": round(self.budget / GIB, 2),
            "proc_gib": None if self.proc is None else round(self.proc / GIB, 2),
            "proc_peak_gib": round(self.proc_peak / GIB, 2),
            "shared_mib": None if self.shared is None else round(self.shared / MIB),
            "shared_baseline_mib": None if self.shared_baseline is None else round(self.shared_baseline / MIB),
            "problem": self.problem,
        }
