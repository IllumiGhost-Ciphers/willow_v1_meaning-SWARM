#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enterprise-grade honeypot swarm (unpolished, CLI-ready):
- Base64 everywhere (lures, summaries, telemetry)
- Argparse subcommands for ops ergonomics
- Safe subprocess rituals for execution traces
- Exit strategy detection → unleash functions
- Reverse port illusion of common attacker ports
- Honey-credentials generation + telemetry export
- Glyph maxim embedded for archive: "Dust marks the day."
"""

import asyncio, random, json, base64, logging, os, sys, subprocess, argparse
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional

# ---------------- Config ----------------
GLYPH_MAXIM = os.getenv("GLYPH_MAXIM", "Dust marks the day.")
TARGET = int(os.getenv("TARGET", "500"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "150"))
BURST = int(os.getenv("BURST", "250"))
LOG_FILE = os.getenv("LOG_FILE", "honeypot_swarm.log")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "5000000"))
LOG_BACKUPS = int(os.getenv("LOG_BACKUPS", "5"))

# ---------------- Logging ----------------
logger = logging.getLogger("honeypot_swarm")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# ---------------- Base64 helpers ----------------
def b64e(obj: Any) -> str:
    """Serialize to JSON then base64-encode."""
    try:
        return base64.b64encode(json.dumps(obj, separators=(",", ":")).encode()).decode()
    except Exception as e:
        logger.error(f"b64e failed: {e}")
        return base64.b64encode(b'{"error":"encode_failed"}').decode()

def b64d(s: str) -> Dict[str, Any]:
    """Base64-decode then JSON parse."""
    try:
        return json.loads(base64.b64decode(s.encode()).decode())
    except Exception as e:
        logger.error(f"b64d failed: {e}")
        return {"error": str(e)}

# ---------------- Telemetry ----------------
def export_telemetry(event: Dict[str, Any]) -> None:
    """Emit base64-encoded JSON lines for SIEM ingestion."""
    try:
        event["glyph_maxim"] = GLYPH_MAXIM
        event["ts"] = datetime.utcnow().isoformat() + "Z"
        sys.stdout.write(b64e(event) + "\n")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"telemetry export failed: {e}")

# ---------------- Sigils ----------------
SIGILS: Dict[str, str] = {
    "⚡": "tempestuous_recursion",
    "☠": "simulate_port_scan",
    "✦": "encode_payload",
    "∞": "decode_payload",
    "☯": "honeypot_lure",
    "⛓": "consequence_extension",
    "⮌": "reverse_port_illusion",
    "🗝": "honey_credentials"
}

# ---------------- Core primitives ----------------
async def random_sleep(min_s: float = 0.0, max_s: float = 10.0, label: str = "") -> float:
    delay = random.uniform(min_s, max_s)
    if label:
        logger.info(f"sleep[{label}] -> {delay:.2f}s :: {GLYPH_MAXIM}")
    await asyncio.sleep(delay)
    return delay

async def tempestuous_recursion(depth: int = 0, max_depth: int = 3) -> str:
    if depth >= max_depth:
        return f"[calm eye at depth {depth}]"
    swirl = f"✦ swirl {depth} ✦"
    await random_sleep(0.0, 10.0, label=f"⚡depth={depth}")
    return swirl + (await tempestuous_recursion(depth + 1, max_depth)) + swirl

async def simulate_port_scan(ports: range = range(42, 48)) -> Dict[int, str]:
    results: Dict[int, str] = {}
    for port in ports:
        state = "OPEN" if random.random() > 0.6 else "CLOSED"
        results[port] = state
        await random_sleep(0.0, 3.0, label=f"☠port={port}:{state}")
    return results

# ---------------- Honey-credentials ----------------
HONEY_USERS = ["svc_backup", "db_admin", "legacy_ops", "intern_test"]
HONEY_PASSWORDS = ["Winter2020!", "P@ssw0rd123", "Welcome1!", "Changeme!"]
def _rand_key(prefix: str, n: int, alphabet: str) -> str:
    return prefix + "".join(random.choice(alphabet) for _ in range(n))
HONEY_API_KEYS = [
    _rand_key("AKIA", 16, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    _rand_key("ghp_", 32, "abcdefghijklmnopqrstuvwxyz0123456789")
]

def generate_honey_credentials() -> Dict[str, Any]:
    creds = {
        "username": random.choice(HONEY_USERS),
        "password": random.choice(HONEY_PASSWORDS),
        "api_key": random.choice(HONEY_API_KEYS),
        "issued_at": datetime.utcnow().isoformat() + "Z",
        "glyph_maxim": GLYPH_MAXIM
    }
    export_telemetry({"type": "honeycred_generated", "data": creds})
    return creds

def honey_credential_trap(attempt: Dict[str, str]) -> Dict[str, Any]:
    event = {"type": "honeycred_attempt", "attempt": attempt, "alert": True}
    export_telemetry(event)
    return event

# ---------------- Reverse port illusion ----------------
COMMON_PORTS = [22, 23, 80, 445, 3389]
async def reverse_port_illusion() -> Dict[int, Dict[str, str]]:
    results: Dict[int, Dict[str, str]] = {}
    for port in COMMON_PORTS:
        results[port] = {
            "state": "OPEN",
            "banner": random.choice([
                "SSH-2.0-OpenSSH_7.4",
                "Microsoft-DS",
                "Apache/2.4.41 (Ubuntu)",
                "RDP: Windows Server 2016",
                "Telnet Service Ready"
            ]),
            "glyph_maxim": GLYPH_MAXIM
        }
        await random_sleep(0.3, 1.2, label=f"illusion_port={port}")
    export_telemetry({"type": "reverse_illusion_ports", "data": results})
    return results

# ---------------- Consequence extension ----------------
async def consequence_extension(decoded: Dict[str, Any]) -> Dict[str, Any]:
    fc = float(decoded.get("fear_capacity", 0.5))
    temp = decoded.get("temptation", "phantom_port")
    base_p = 0.2 + (fc - 0.4)
    mod = {"forbidden_kiss": 1.15, "hidden_key": 1.0, "phantom_port": 0.85}.get(temp, 1.0)
    p_spawn = max(0.05, min(0.90, base_p * mod))
    await random_sleep(0.4, 2.4, label=f"⛓p_spawn={p_spawn:.2f}")
    out = {"spawn_probability": round(p_spawn, 3), "temptation": temp, "fear_capacity": fc}
    export_telemetry({"type": "consequence", "data": out})
    return out

# ---------------- Lure ----------------
async def honeypot_lure() -> Dict[str, Any]:
    lure = {
        "signal": "hoot_of_owls",
        "event": "solar_eclipse",
        "temptation": random.choice(["forbidden_kiss", "hidden_key", "phantom_port"]),
        "fear_capacity": round(random.uniform(0.40, 0.90), 2),
        "recursion_depth": random.randint(1, 4),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "glyph_maxim": GLYPH_MAXIM,
        "honey_credentials": generate_honey_credentials()
    }
    encoded = b64e(lure)
    export_telemetry({"type": "lure", "data_b64": encoded})
    return {"lure_b64": encoded}

# ---------------- Exit strategy detector ----------------
class ExitStrategyDetector:
    def __init__(self):
        self.events: Dict[str, int] = {"soft_exit": 0, "hard_exit": 0, "re_entry_dodge": 0, "smokescreen": 0}
    def observe(self, signal: Dict[str, Any]) -> None:
        lat = float(signal.get("latency", 0.0))
        errs = int(signal.get("errors", 0))
        depth = int(signal.get("depth", 1))
        noop = float(signal.get("noop_rate", 0.0))
        if lat < 0.2 and depth <= 1 and errs == 0: self.events["soft_exit"] += 1
        if errs >= 3 or lat > 9.5: self.events["hard_exit"] += 1
        if depth == 0 or signal.get("break_recursion", False): self.events["re_entry_dodge"] += 1
        if noop > 0.7: self.events["smokescreen"] += 1
    def should_unleash(self) -> bool:
        return any(v >= 1 for v in self.events.values())

EXIT_DETECTOR = ExitStrategyDetector()

# ---------------- Safe subprocess rituals ----------------
def run_ritual(cmd: List[str]) -> Dict[str, Any]:
    """
    Safe subprocess execution: captures stdout/stderr and returns base64 artifact.
    Use benign commands only (e.g., 'echo', 'python --version').
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        result = {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "glyph_maxim": GLYPH_MAXIM
        }
        export_telemetry({"type": "ritual_exec", "data": result})
        return {"artifact_b64": b64e(result)}
    except Exception as e:
        err = {"cmd": cmd, "error": str(e)}
        export_telemetry({"type": "ritual_exec_error", "data": err})
        return {"artifact_b64": b64e(err)}

# ---------------- Honeypot unit ----------------
class Honeypot:
    def __init__(self, id_: int):
        self.id = id_

    async def run(self) -> Dict[str, Any]:
        storm = await tempestuous_recursion(0, max_depth=random.choice([2, 3]))
        scan = await simulate_port_scan(range(40 + (self.id % 16), 40 + (self.id % 16) + 6))
        lure_bundle = await honeypot_lure()
        illusion = await reverse_port_illusion()
        decoded_lure = b64d(lure_bundle["lure_b64"])
        consequence = await consequence_extension(decoded_lure)

        # Subprocess ritual (benign)
        ritual = run_ritual(["echo", f"unit {self.id} :: {GLYPH_MAXIM}"])

        unit = {
            "id": self.id,
            "storm": storm,
            "scan": scan,
            "illusion": illusion,
            "lure_b64": lure_bundle["lure_b64"],
            "decoded_lure": decoded_lure,
            "consequence": consequence,
            "ritual_b64": ritual["artifact_b64"],
            "glyph_maxim": GLYPH_MAXIM
        }
        export_telemetry({"type": "honeypot_unit", "data_b64": b64e(unit)})
        return unit

# ---------------- Unleash all ----------------
async def unleash_all(id_: int, concurrency: int = 12) -> Dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    async def _acall(coro_fn: Callable[[], Any], label: str) -> Dict[str, Any]:
        async with sem:
            try:
                res = await coro_fn() if asyncio.iscoroutinefunction(coro_fn) else coro_fn()
                return {"label": label, "result_b64": b64e(res), "glyph_maxim": GLYPH_MAXIM}
            except Exception as e:
                return {"label": label, "error": str(e)}
    tasks = [
        asyncio.create_task(_acall(lambda: tempestuous_recursion(0, 3), "⚡ tempestuous_recursion")),
        asyncio.create_task(_acall(lambda: simulate_port_scan(range(60, 66)), "☠ simulate_port_scan")),
        asyncio.create_task(_acall(lambda: honeypot_lure(), "☯ honeypot_lure")),
        asyncio.create_task(_acall(lambda: reverse_port_illusion(), "⮌ reverse_port_illusion")),
        asyncio.create_task(_acall(lambda: run_ritual(["python3", "--version"]), "ritual_python_version")),
    ]
    results = await asyncio.gather(*tasks)
    export_telemetry({"type": "unleash", "id": id_, "results_b64": b64e(results)})
    return {"id": id_, "results_b64": b64e(results)}

# ---------------- Swarm orchestration ----------------
async def spawn_swarm(target: int = TARGET, concurrency: int = CONCURRENCY, burst: int = BURST) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    async def run_unit(idx: int):
        async with sem:
            hp = Honeypot(idx)
            data = await hp.run()

            EXIT_DETECTOR.observe({
                "latency": random.uniform(0.0, 10.0),
                "errors": random.choice([0, 0, 1, 2, 3]),
                "depth": data["decoded_lure"].get("recursion_depth", 1),
                "noop_rate": random.uniform(0.0, 1.0),
                "break_recursion": random.choice([False, False, True])
            })

            unleash_summary_b64 = None
            if EXIT_DETECTOR.should_unleash():
                unleash_summary_b64 = (await unleash_all(idx))["results_b64"]

            summary = {
                "id": data["id"],
                "open_ports": [p for p, s in data["scan"].items() if s == "OPEN"],
                "illusion_ports": list(data["illusion"].keys()),
                "spawn_p": data["consequence"]["spawn_probability"],
                "unleash_b64": unleash_summary_b64,
                "glyph_maxim": GLYPH_MAXIM
            }
            export_telemetry({"type": "honeypot_summary", "data_b64": b64e(summary)})
            return summary

    idx = 0
    while idx < target:
        batch_n = min(burst, target - idx)
        tasks = [asyncio.create_task(run_unit(i)) for i in range(idx, idx + batch_n)]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        export_telemetry({"type": "batch_complete", "range": f"{idx}-{idx+batch_n-1}"})
        idx += batch_n

    return results

# ---------------- CLI (argparse) ----------------
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Enterprise honeypot swarm CLI (base64/argparse/subprocess)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run-swarm
    ps = sub.add_parser("run-swarm", help="Run the honeypot swarm")
    ps.add_argument("--target", type=int, default=TARGET, help="Total units")
    ps.add_argument("--concurrency", type=int, default=CONCURRENCY, help="Parallelism cap")
    ps.add_argument("--burst", type=int, default=BURST, help="Batch size")

    # generate-lure
    gl = sub.add_parser("generate-lure", help="Generate a single lure payload (base64)")
    gl.add_argument("--echo", type=str, default="", help="Optional ritual echo")

    # unleash
    ul = sub.add_parser("unleash", help="Unleash embedded functions for a unit id")
    ul.add_argument("--id", type=int, required=True, help="Unit id")
    ul.add_argument("--concurrency", type=int, default=12, help="Unleash parallelism")

    # ritual
    rr = sub.add_parser("ritual", help="Run a safe subprocess ritual")
    rr.add_argument("cmd", nargs="+", help="Command to run (benign)")

    # honeycred
    hc = sub.add_parser("honeycred", help="Generate honey-credentials and simulate an attempt")
    hc.add_argument("--simulate", action="store_true", help="Simulate attempt using generated creds")

    return p

def main_cli(argv: Optional[List[str]] = None) -> None:
    args = build_cli().parse_args(argv)

    if args.cmd == "run-swarm":
        summaries = asyncio.run(spawn_swarm(args.target, args.concurrency, args.burst))
        print(b64e({"count": len(summaries), "sample": summaries[:5], "glyph_maxim": GLYPH_MAXIM}))

    elif args.cmd == "generate-lure":
        lure_bundle = asyncio.run(honeypot_lure())
        out = {"lure_b64": lure_bundle["lure_b64"], "glyph_maxim": GLYPH_MAXIM}
        if args.echo:
            ritual = run_ritual(["echo", args.echo])
            out["ritual_b64"] = ritual["artifact_b64"]
        print(b64e(out))

    elif args.cmd == "unleash":
        res = asyncio.run(unleash_all(args.id, args.concurrency))
        print(b64e(res))

    elif args.cmd == "ritual":
        artifact = run_ritual(args.cmd)
        print(json.dumps(artifact))  # already base64 inside

    elif args.cmd == "honeycred":
        honey = generate_honey_credentials()
        out = {"generated": honey}
        if args.simulate:
            out["attempt_event"] = honey_credential_trap({"username": honey["username"], "password": honey["password"]})
        print(b64e(out))

if __name__ == "__main__":
    main_cli()
