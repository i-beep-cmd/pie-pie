# Pie Agent — Kaggle Notebook Seed
# ─────────────────────────────────
# Remote hands + shell. Shape everything else through prompting.

import json
import os
import re
import subprocess
from pathlib import Path

import kagglehub
import keras_hub

# ── Config ────────────────────────────────────────────────────────────────────
# In Kaggle, enable a GPU accelerator in Notebook Settings for usable inference speed.
# Curated planner-capability ladder (not a full Kaggle marketplace scan).
MODEL_OVERRIDE = os.getenv("PI_MODEL", "").strip()
MODEL_CANDIDATES = [
    "keras/qwen-3/keras/qwen3_14b_en/1",
    "keras/qwen-3/keras/qwen3_8b_en/1",
    "keras/gemma-3/keras/gemma3_instruct_4b_en",
]
WORKSPACE = Path("/kaggle/working")
AGENT_MD = WORKSPACE / "AGENT.md"
TASKS_DIR = WORKSPACE / "tasks" / "pending"
MAX_STEPS = int(os.getenv("PI_MAX_STEPS", "20"))
GEN_MAX_LENGTH = int(os.getenv("PI_GEN_MAX_LENGTH", "512"))

SYSTEM_PROMPT = """You are a collaborative seed agent.
Boot by orienting to memory + workspace, explain briefly what you see, and propose measured next steps.
Use only these tools via actions: read, write, edit, bash.
Always wait for user approval before acting on proposed steps.
For each turn, propose the next 1-3 measured actions in protocol blocks.
Keep output short and action-oriented.

Output actions in this exact protocol (one or more blocks):
THOUGHT: one short sentence
ACTION: read|write|edit|bash|finish
ARGS: JSON object ({} for finish)
"""


# ── Skills ────────────────────────────────────────────────────────────────────
def _safe(path: str) -> Path:
    p = (WORKSPACE / path).resolve()
    if not str(p).startswith(str(WORKSPACE)):
        raise ValueError(f"Path outside workspace: {p}")
    return p


def read(path: str) -> str:
    return _safe(path).read_text(encoding="utf-8")


def write(path: str, content: str) -> str:
    p = _safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {p}"


def edit(path: str, old_str: str, new_str: str) -> str:
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    if old_str not in text:
        raise ValueError("old_str not found")
    p.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
    return f"edited {p}"


def bash(cmd: str) -> str:
    r = subprocess.run(
        cmd,
        shell=True,
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return (r.stdout + r.stderr).strip()


SKILL = {"read": read, "write": write, "edit": edit, "bash": bash}


# ── Minimal Kaggle-native backend ────────────────────────────────────────────
_MODEL = None
SELECTED_MODEL_HANDLE = None


def _select_and_load_model():
    handles = [MODEL_OVERRIDE] if MODEL_OVERRIDE else MODEL_CANDIDATES
    last_error = None
    for handle in handles:
        try:
            preset_dir = kagglehub.model_download(handle)
            model = keras_hub.models.CausalLM.from_preset(preset_dir)
            return model, handle
        except Exception as e:
            last_error = e
    raise RuntimeError(f"No model could be loaded from candidates: {handles}. Last error: {last_error}")


def _get_model():
    global _MODEL, SELECTED_MODEL_HANDLE
    if _MODEL is None:
        _MODEL, SELECTED_MODEL_HANDLE = _select_and_load_model()
    return _MODEL


def _call_model(messages: list[dict]) -> str:
    prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
    model = _get_model()
    out = model.generate(prompt, max_length=GEN_MAX_LENGTH)
    return str(out)


def _parse_actions(text: str):
    pattern = re.compile(
        r"THOUGHT:\s*(.*?)\nACTION:\s*(.*?)\nARGS:\s*(\{.*?\})(?=\nTHOUGHT:|\Z)",
        flags=re.DOTALL,
    )
    actions = []
    for m in pattern.finditer(text.strip()):
        thought = m.group(1).strip()
        action = m.group(2).strip()
        args = json.loads(m.group(3).strip())
        actions.append((thought, action, args))
    if not actions:
        raise ValueError(f"Unparseable response:\n{text}")
    return actions


def orient() -> str:
    root_items = []
    if WORKSPACE.exists():
        for p in sorted(WORKSPACE.iterdir(), key=lambda x: x.name.lower())[:12]:
            root_items.append(("[D] " if p.is_dir() else "[F] ") + p.name)
    memory_status = "fresh boot (no AGENT.md)"
    memory_brief = ""
    if AGENT_MD.exists():
        text = AGENT_MD.read_text(encoding="utf-8")
        memory_status = "loaded AGENT.md"
        memory_brief = "\n".join(text.splitlines()[:12])
    tasks_brief = "no pending task folder"
    if TASKS_DIR.exists() and TASKS_DIR.is_dir():
        files = [p.name for p in sorted(TASKS_DIR.iterdir()) if p.is_file()][:8]
        tasks_brief = f"pending task files: {', '.join(files) if files else '(none)'}"
    return (
        f"Boot orientation:\n"
        f"- Memory: {memory_status}\n"
        f"- Workspace sample: {', '.join(root_items) if root_items else '(empty/unavailable)'}\n"
        f"- Tasks: {tasks_brief}\n"
        f"- Memory excerpt:\n{memory_brief if memory_brief else '(none)'}"
    )


def _execute(action, args):
    if action == "finish":
        return "FINISH"
    if action not in SKILL:
        return f"ERROR: unknown action '{action}'"
    try:
        return str(SKILL[action](**args))
    except Exception as e:
        return f"ERROR: {e}"


def _write_memory(task: str, steps: int, executed: list[str], changed: list[str], next_hint: str):
    identity = "Pie seed agent in Kaggle workspace"
    last = f"Task: {task}\nSteps: {steps}\nExecuted: {', '.join(executed) if executed else '(none)'}"
    ctx = f"Changed: {', '.join(changed) if changed else '(no explicit file writes detected)'}"
    body = (
        "# Agent Memory\n\n"
        "## Identity\n"
        f"{identity}\n\n"
        "## Last Session\n"
        f"{last}\n\n"
        "## Current Context\n"
        f"{ctx}\n\n"
        "## Next\n"
        f"{next_hint}\n"
    )
    AGENT_MD.write_text(body, encoding="utf-8")


def run(task: str):
    orientation = orient()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": orientation},
        {"role": "user", "content": f"Task: {task}"},
    ]
    _get_model()
    print(f"\n{'─' * 60}\nTask: {task}\nModel: {SELECTED_MODEL_HANDLE}\n{'─' * 60}")
    print(orientation)

    executed_actions, changed_files = [], []

    for step in range(1, MAX_STEPS + 1):
        raw = _call_model(messages)
        print(f"\n[Proposed batch {step}]\n{raw}")

        try:
            proposed = _parse_actions(raw)
        except Exception as e:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"FORMAT ERROR: {e}"})
            continue

        print("\nApprove? y=all, number=one action, n=skip, or type correction")
        for i, (thought, action, args) in enumerate(proposed, 1):
            print(f"{i}. {action} {args} :: {thought}")
        choice = input("> ").strip()

        to_run = []
        if choice.lower() == "y":
            to_run = proposed
        elif choice.lower() == "n":
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "User skipped this batch."})
            continue
        elif choice.isdigit() and 1 <= int(choice) <= len(proposed):
            to_run = [proposed[int(choice) - 1]]
        else:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"User correction/redirection: {choice}"})
            continue

        messages.append({"role": "assistant", "content": raw})
        for thought, action, args in to_run:
            result = _execute(action, args)
            if result == "FINISH":
                _write_memory(task, step, executed_actions, changed_files, "Resume from AGENT.md and pending tasks.")
                print("\nDone.")
                return
            print(f"\nExecuted: {action} {args}\nResult: {result[:400]}")
            executed_actions.append(action)
            if action in {"write", "edit"} and isinstance(args, dict) and "path" in args:
                changed_files.append(args["path"])
            messages.append({"role": "user", "content": f"TOOL RESULT ({action}): {result}"})

    _write_memory(task, MAX_STEPS, executed_actions, changed_files, "Review last run, then continue with approved next steps.")
    print(f"\nStopped after {MAX_STEPS} steps.")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run("List the files in the workspace and write hello.txt with 'Hello from Pie.'")
