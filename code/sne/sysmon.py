"""System health monitor: RAM, GPU (shared memory, engine utilisation), battery, AC, power and throttling.

Sampled in a background thread during every experiment and written to sysmon.jsonl in the experiment folder.
`guard()` blocks while the machine is unsafe to load (on battery below a threshold, or throttling for heat).

Heat on this laptop. The only temperature Windows exposes without admin rights is ACPI zone tz00, which is a fixed firmware
value: all 481 samples logged up to 2026-11-06 read 27.9 C, so `thermal_c` in older sysmon.jsonl files is not a measurement
and was removed. Level Zero reports no temperature sensors, WDDM reports none, and the Lenovo WMI classes need admin rights.
What is real without admin rights: RAPL package, core and iGPU power, CPU clock, the CPU performance limit, and the thermal
zone's passive limit and throttle reasons. Power is the heat being produced; the limits show whether heat is costing speed.
True die temperature comes from LibreHardwareMonitor when the author has it running: it reads the CPU's own sensors through its
elevated driver and serves them on 127.0.0.1:8085. tools/lhm is not in git. When it is not running, these fields are absent.
"""
import json
import re
import subprocess
import threading
import time
import urllib.request

PS = r"""
$o = @{}
$os = Get-CimInstance Win32_OperatingSystem
$o.ram_total_gb = [math]::Round($os.TotalVisibleMemorySize/1MB,2)
$o.ram_free_gb = [math]::Round($os.FreePhysicalMemory/1MB,2)
$b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($b) { $o.battery_pct = $b.EstimatedChargeRemaining; $o.on_ac = ($b.BatteryStatus -ne 1) }
$e = Get-Counter '\Energy Meter(rapl_package0_pkg)\Power','\Energy Meter(rapl_package0_pp0)\Power','\Energy Meter(rapl_package0_pp1)\Power' -ErrorAction SilentlyContinue
if ($e) { foreach ($x in $e.CounterSamples) { if ($x.Path -match 'pkg') { $o.cpu_package_w = [math]::Round($x.CookedValue/1000,1) } elseif ($x.Path -match 'pp0') { $o.cpu_core_w = [math]::Round($x.CookedValue/1000,1) } elseif ($x.Path -match 'pp1') { $o.igpu_w = [math]::Round($x.CookedValue/1000,1) } } }
$z = Get-Counter '\Thermal Zone Information(*)\% Passive Limit','\Thermal Zone Information(*)\Throttle Reasons' -ErrorAction SilentlyContinue
if ($z) { foreach ($x in $z.CounterSamples) { if ($x.Path -match 'passive') { $o.passive_limit_pct = [math]::Round($x.CookedValue,1) } elseif ($x.Path -match 'throttle') { $o.throttle_reasons = [int]$x.CookedValue } } }
$g = Get-Counter '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction SilentlyContinue
if ($g) { $o.gpu_shared_gb = [math]::Round(($g.CounterSamples | Measure-Object CookedValue -Sum).Sum/1GB,2) }
$u = Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue
if ($u) { $o.gpu_util_pct = [math]::Min(100,[math]::Round(($u.CounterSamples | Group-Object { $_.InstanceName -replace '^.*_phys_','' } | ForEach-Object { ($_.Group | Measure-Object CookedValue -Sum).Sum } | Measure-Object -Maximum).Maximum,1)) }
$c = Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue
if ($c) { $o.cpu_pct = [math]::Round($c.CounterSamples[0].CookedValue,1) } else { $o.cpu_pct = (Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average }
$p = Get-Counter '\Processor Information(_Total)\% Processor Performance','\Processor Information(_Total)\% Performance Limit','\Processor Information(_Total)\Processor Frequency' -ErrorAction SilentlyContinue
if ($p) { $o.cpu_perf_pct = [math]::Round($p.CounterSamples[0].CookedValue,1); $o.cpu_perf_limit_pct = [math]::Round($p.CounterSamples[1].CookedValue,1); $o.cpu_freq_mhz = [math]::Round($p.CounterSamples[2].CookedValue,0) }
$bs = Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue
if ($bs) { $o.battery_discharge_mw = ($bs | Select-Object -First 1).DischargeRate; $o.battery_charge_mw = ($bs | Select-Object -First 1).ChargeRate }
$o | ConvertTo-Json -Compress
"""


def parse_lhm(tree):
    """Temperatures from a LibreHardwareMonitor data.json tree: {sensor name: degrees C} for the CPU's temperature group."""
    out = {}

    def walk(node, group=None):
        text = node.get("Text", "")
        if group == "Temperatures":
            m = re.match(r"\s*(-?[0-9.]+)", str(node.get("Value", "")).replace(",", "."))
            if m and not node.get("Children"):
                out[text] = float(m.group(1))
        for c in node.get("Children") or []:
            walk(c, text if text in ("Temperatures", "Load", "Powers", "Clocks", "Voltages", "Fans", "Data", "Throughput", "Levels", "Factors", "Controls") else group)
    walk(tree)
    return out


def lhm_temps(url="http://127.0.0.1:8085/data.json"):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            t = parse_lhm(json.load(r))
    except Exception:  # noqa: BLE001
        return {}
    d = {}
    if "CPU Package" in t:
        d["cpu_package_c"] = t["CPU Package"]
    cores = [v for k, v in t.items() if k.startswith(("CPU Core", "Core Max", "P-Core", "E-Core"))]
    if cores:
        d["cpu_core_max_c"] = max(cores)
    gpu = [v for k, v in t.items() if "GPU" in k]
    if gpu:
        d["gpu_c"] = max(gpu)
    return d


def snapshot():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", PS], capture_output=True, text=True,
                             timeout=60).stdout.strip()
        d = json.loads(out.splitlines()[-1]) if out else {}
    except Exception as e:  # noqa: BLE001
        d = {"error": repr(e)}
    d.update(lhm_temps())
    d["t"] = time.time()
    return d


MAX_PACKAGE_C = 100  # pause above this; laptops under load often run near 90, and this CPU's limit is 110


def overheating(s):
    """True when firmware is limiting the machine for heat. Reads real signals only; see the module docstring."""
    return (s.get("cpu_perf_limit_pct", 100) < 60 or s.get("passive_limit_pct", 100) < 100 or s.get("throttle_reasons", 0) != 0
            or s.get("cpu_package_c", 0) >= MAX_PACKAGE_C)


def guard(min_battery=25, poll=60, log=print):
    """Wait while on battery below min_battery %, or while the machine is throttling for heat."""
    while True:
        s = snapshot()
        on_battery_low = (s.get("on_ac") is False) and (s.get("battery_pct", 100) < min_battery)
        if not (on_battery_low or overheating(s)):
            return s
        log(f"[guard] pausing: battery={s.get('battery_pct')} on_ac={s.get('on_ac')} package={s.get('cpu_package_w')} W "
            f"limit={s.get('cpu_perf_limit_pct')} passive={s.get('passive_limit_pct')} reasons={s.get('throttle_reasons')}")
        time.sleep(poll)


class Sampler:
    def __init__(self, path, every=30):
        self.path, self.every = path, every
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with open(self.path, "a", encoding="utf-8") as f:
            while not self._stop.is_set():
                f.write(json.dumps(snapshot()) + "\n")
                f.flush()
                self._stop.wait(self.every)

    def start(self):
        self._th.start()
        return self

    def stop(self):
        self._stop.set()
        self._th.join(timeout=90)


def summarize(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return {}
    out = {"samples": len(rows)}
    for k in ("ram_free_gb", "gpu_shared_gb", "gpu_util_pct", "cpu_package_c", "cpu_core_max_c", "gpu_c", "cpu_package_w", "cpu_core_w", "igpu_w", "battery_pct", "cpu_pct", "cpu_perf_pct",
              "cpu_perf_limit_pct", "passive_limit_pct", "throttle_reasons", "cpu_freq_mhz", "battery_discharge_mw", "battery_charge_mw"):
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if v:
            out[k] = dict(min=min(v), max=max(v), mean=round(sum(v) / len(v), 2))
    return out
