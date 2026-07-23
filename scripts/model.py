#!/usr/bin/env python3
"""Model registry resolver — the bridge between scripts/models.toml and the harness.

Two consumers:
  * bash scripts `eval "$(model.py env <name>)"` to get CONTAINER / SOURCE / ARCH /
    PROMPT_TOKENS / CONVERT_ENV / NOTES as shell variables (paths already resolved
    against models_root / COLI_MODELS_ROOT).
  * python tools `from model import resolve; m = resolve("minimax-m3")`.

Keeping path resolution and prompt expansion in ONE place is the whole point: a new
model is a registry block, never a code edit in five different scripts.

Usage:
  model.py list                 # names + notes
  model.py env   <name>         # shell assignments (for `eval` in bash)
  model.py get   <name> <field> # one resolved field (container/source/arch/prompt/notes)
  model.py path  <name>         # resolved container path (shorthand for `get <name> container`)
"""
import os
import shlex
import sys
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("model.py: needs Python 3.11+ (tomllib). Found %s" % sys.version.split()[0])

REGISTRY = Path(__file__).resolve().parent / "models.toml"


def _load():
    if not REGISTRY.exists():
        sys.exit(f"model.py: registry not found: {REGISTRY}")
    with open(REGISTRY, "rb") as f:
        return tomllib.load(f)


def _root(reg):
    # COLI_MODELS_ROOT wins so the same registry works on any host.
    return os.environ.get("COLI_MODELS_ROOT") or reg.get("models_root") or "."


def _join(root, p):
    return p if p.startswith("/") else str(Path(root) / p)


def _expand_prompt(spec):
    """"A..B" -> "A A+1 ... B" (inclusive); otherwise pass through as an id list."""
    spec = str(spec).strip()
    if ".." in spec:
        a, b = spec.split("..", 1)
        return " ".join(str(i) for i in range(int(a), int(b) + 1))
    return spec


def registry_models():
    reg = _load()
    return {k: v for k, v in reg.items() if isinstance(v, dict)}


def resolve(name):
    reg = _load()
    models = {k: v for k, v in reg.items() if isinstance(v, dict)}
    if name not in models:
        avail = ", ".join(sorted(models)) or "(none)"
        sys.exit(f"model.py: unknown model '{name}'. Known: {avail}")
    m = models[name]
    root = _root(reg)
    return {
        "name": name,
        "arch": m.get("arch", ""),
        "container": _join(root, m.get("container", "")) if m.get("container") else "",
        "source": _join(root, m.get("source", "")) if m.get("source") else "",
        "prompt": _expand_prompt(m.get("prompt", "1")),
        "prompt_spec": str(m.get("prompt", "1")),
        "notes": m.get("notes", ""),
        "convert_env": {str(k): str(v) for k, v in (m.get("convert_env") or {}).items()},
    }


def _cmd_list():
    for name, m in sorted(registry_models().items()):
        print(f"{name:16s}  {m.get('notes', '')}")


def _cmd_env(name):
    r = resolve(name)
    convert_env = " ".join(f"{k}={v}" for k, v in r["convert_env"].items())
    out = {
        "COLI_MODEL": r["name"],
        "ARCH": r["arch"],
        "CONTAINER": r["container"],
        "SOURCE": r["source"],
        "PROMPT_TOKENS": r["prompt"],
        "PROMPT_SPEC": r["prompt_spec"],
        "CONVERT_ENV": convert_env,
        "NOTES": r["notes"],
    }
    for k, v in out.items():
        print(f"{k}={shlex.quote(str(v))}")


def _cmd_get(name, field):
    r = resolve(name)
    if field not in r:
        sys.exit(f"model.py: unknown field '{field}'. Fields: {', '.join(r)}")
    v = r[field]
    print(" ".join(f"{k}={vv}" for k, vv in v.items()) if isinstance(v, dict) else v)


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    cmd = argv[1]
    if cmd == "list":
        _cmd_list()
    elif cmd == "env" and len(argv) == 3:
        _cmd_env(argv[2])
    elif cmd == "get" and len(argv) == 4:
        _cmd_get(argv[2], argv[3])
    elif cmd == "path" and len(argv) == 3:
        print(resolve(argv[2])["container"])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
