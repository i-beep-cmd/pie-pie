#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import ollama

MODEL_NAME = "gemma4:e2b"
PI_DIR = ".pi"
CORE_DIR = ".pi/core"
SNAPSHOT_DIR = ".pi/core/snapshots"
SHELL_DIR = ".pi/shell"
EXTENSIONS_DIR = ".pi/shell/extensions"
AGENTS_FILE = ".pi/core/AGENTS.md"

DEFAULT_AGENTS_MD = """# PI Agent Identity and Policy

You are PI, a minimalist local coding agent.

## Single Folder Seed Philosophy
- `.pi/core/` holds identity, policy, and snapshots.
- `.pi/shell/` is the writable implementation workspace.
- Extensions should be seeded under `.pi/shell/extensions/`.

## Required Workflow (Must Not Be Skipped or Reordered)
1. Plan
2. Snapshot
3. Implement
4. Verify

The agent must never skip phases and must never reorder phases.

## Code Placement Rule
All agent-created code belongs in `.pi/shell/` unless the Python runtime itself is intentionally being changed by the human.
"""

SYSTEM_PROMPT = """You are PI, a local coding agent running in a single-folder-seed workspace.

Operating model:
- Core: `.pi/core/` for identity, policy, snapshots (protected).
- Shell: `.pi/shell/` for writable implementation work.
- Extensions: place self-extensions in `.pi/shell/extensions/`.

You must follow this workflow exactly and in order:
1) Plan
2) Snapshot
3) Implement
4) Verify

Rules:
- No mutation before a successful snapshot save.
- Use tools for filesystem and shell actions.
- Keep plans concise and explicit.
- During the Plan phase, your planning response must begin with the exact prefix `Plan:`.
- Prefer creating and editing files in `.pi/shell/`.
- Treat `.pi/core/` as protected policy/state area.
- If starting a new task block, begin again at Plan then Snapshot before mutation.
"""


@dataclass
class SessionState:
    phase: str = "plan"
    plan_emitted_this_cycle: bool = False
    snapshot_saved_this_cycle: bool = False
    implementation_started: bool = False

    def begin_new_task_block(self) -> None:
        self.phase = "plan"
        self.plan_emitted_this_cycle = False
        self.snapshot_saved_this_cycle = False
        self.implementation_started = False

    def mark_plan_emitted(self) -> None:
        self.plan_emitted_this_cycle = True
        if self.phase == "plan":
            self.phase = "snapshot"

    def mark_snapshot_saved(self) -> None:
        self.snapshot_saved_this_cycle = True
        self.phase = "implement"

    def mark_mutation(self) -> None:
        self.implementation_started = True
        if self.phase in {"plan", "snapshot"}:
            self.phase = "implement"

    def mark_verify(self) -> None:
        if self.phase == "implement":
            self.phase = "verify"


class PiAgent:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.pi_dir = self.root / PI_DIR
        self.core_dir = self.root / CORE_DIR
        self.snapshot_dir = self.root / SNAPSHOT_DIR
        self.shell_dir = self.root / SHELL_DIR
        self.extensions_dir = self.root / EXTENSIONS_DIR
        self.extensions_registry_file = self.core_dir / "extensions.json"
        self.agents_file = self.root / AGENTS_FILE
        self.state = SessionState()
        self.allow_runtime_writes = os.environ.get("PI_ALLOW_RUNTIME_WRITES", "0") == "1"

    def init_layout(self) -> None:
        for d in [self.pi_dir, self.core_dir, self.snapshot_dir, self.shell_dir, self.extensions_dir]:
            d.mkdir(parents=True, exist_ok=True)
        if not self.agents_file.exists():
            self.agents_file.write_text(DEFAULT_AGENTS_MD, encoding="utf-8")
        if not self.extensions_registry_file.exists():
            self.extensions_registry_file.write_text("[]\n", encoding="utf-8")

    def read_agents_md(self) -> str:
        try:
            return self.agents_file.read_text(encoding="utf-8")
        except Exception as exc:
            return f"ERROR: could not read {AGENTS_FILE}: {exc}"

    def _resolve_path(self, path_str: str) -> Tuple[Optional[Path], Optional[str]]:
        try:
            path = Path(path_str)
            candidate = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        except Exception as exc:
            return None, f"ERROR: invalid path '{path_str}': {exc}"
        if not self._is_within(candidate, self.root):
            return None, f"ERROR: path outside project root is not allowed: {candidate}"
        return candidate, None

    @staticmethod
    def _is_within(path: Path, base: Path) -> bool:
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False

    def _is_writable_zone(self, target: Path) -> bool:
        if self._is_within(target, self.shell_dir):
            return True
        if self.allow_runtime_writes:
            core_protected = self._is_within(target, self.core_dir)
            return not core_protected and self._is_within(target, self.root)
        return False

    def read(self, path: str) -> str:
        resolved, err = self._resolve_path(path)
        if err:
            return err
        try:
            if resolved.is_dir():
                items = []
                for p in sorted(resolved.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                    kind = "[D]" if p.is_dir() else "[F]"
                    items.append(f"{kind} {p.name}")
                return "\n".join(items)
            return resolved.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"ERROR: file not found: {resolved}"
        except UnicodeDecodeError:
            return f"ERROR: file is not UTF-8 text: {resolved}"
        except Exception as exc:
            return f"ERROR: failed to read {resolved}: {exc}"

    def write(self, path: str, content: str) -> str:
        if not self.state.snapshot_saved_this_cycle:
            return "ERROR: mutation blocked. You must call snapshot('save', <name>) after planning and before write/edit/bash."
        resolved, err = self._resolve_path(path)
        if err:
            return err
        if not self._is_writable_zone(resolved):
            return "ERROR: write blocked by policy. Writes are only allowed under .pi/shell/ by default."
        try:
            is_extension_file = self._is_within(resolved, self.extensions_dir)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            if is_extension_file and resolved.is_file():
                self._register_extension(resolved)
            self.state.mark_mutation()
            return f"OK: wrote {resolved}"
        except Exception as exc:
            return f"ERROR: failed to write {resolved}: {exc}"

    def edit(self, path: str, old_str: str, new_str: str) -> str:
        if not self.state.snapshot_saved_this_cycle:
            return "ERROR: mutation blocked. You must call snapshot('save', <name>) after planning and before write/edit/bash."
        resolved, err = self._resolve_path(path)
        if err:
            return err
        if not self._is_writable_zone(resolved):
            return "ERROR: edit blocked by policy. Edits are only allowed under .pi/shell/ by default."
        try:
            text = resolved.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"ERROR: file not found: {resolved}"
        except Exception as exc:
            return f"ERROR: failed to read for edit {resolved}: {exc}"

        if old_str not in text:
            return "ERROR: old_str not found in file. No changes made."

        replaced = text.replace(old_str, new_str)
        count = text.count(old_str)
        try:
            resolved.write_text(replaced, encoding="utf-8")
            if self._is_within(resolved, self.extensions_dir):
                self._register_extension(resolved)
            self.state.mark_mutation()
            return f"OK: edited {resolved} (replaced {count} occurrence(s))."
        except Exception as exc:
            return f"ERROR: failed to write edited file {resolved}: {exc}"

    def bash(self, cmd: str) -> str:
        if not self.state.snapshot_saved_this_cycle:
            return "ERROR: mutation/check blocked. You must call snapshot('save', <name>) after planning and before bash."
        timeout_seconds = 25
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.root),
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            self.state.mark_verify()
            payload = {
                "status": "ok" if proc.returncode == 0 else "error",
                "exit_code": proc.returncode,
                "stdout": self._truncate_output(proc.stdout),
                "stderr": self._truncate_output(proc.stderr),
                "timed_out": False,
            }
            return json.dumps(payload, ensure_ascii=False)
        except subprocess.TimeoutExpired as exc:
            payload = {
                "status": "timeout",
                "exit_code": None,
                "stdout": self._truncate_output(exc.stdout if isinstance(exc.stdout, str) else ""),
                "stderr": self._truncate_output(exc.stderr if isinstance(exc.stderr, str) else ""),
                "timed_out": True,
                "timeout_seconds": timeout_seconds,
            }
            return json.dumps(payload, ensure_ascii=False)
        except Exception as exc:
            payload = {
                "status": "error",
                "exit_code": None,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
            }
            return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _truncate_output(text: str, max_chars: int = 12000) -> str:
        if len(text) <= max_chars:
            return text
        clipped = len(text) - max_chars
        return f"{text[:max_chars]}\n...<truncated {clipped} chars>..."

    def list_snapshots(self) -> str:
        try:
            snapshots = []
            for archive in self.snapshot_dir.glob("*.tar.gz"):
                stat = archive.stat()
                snapshots.append(
                    {
                        "filename": archive.name,
                        "modified_time": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                        "size_bytes": stat.st_size,
                    }
                )
            snapshots.sort(key=lambda x: x["modified_time"], reverse=True)
            return json.dumps(snapshots, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": f"failed to list snapshots: {exc}"}, ensure_ascii=False)

    def tree(self, path: str = SHELL_DIR, max_depth: int = 3, format: str = "text") -> str:
        resolved, err = self._resolve_path(path or SHELL_DIR)
        if err:
            return err
        if not resolved.exists():
            return f"ERROR: path does not exist: {resolved}"
        if not self._is_within(resolved, self.root):
            return f"ERROR: path outside project root is not allowed: {resolved}"
        try:
            depth_value = int(max_depth)
        except (TypeError, ValueError):
            return "ERROR: max_depth must be an integer."
        max_depth = max(0, min(depth_value, 8))
        fmt = (format or "text").strip().lower()
        if fmt not in {"text", "json"}:
            return "ERROR: tree format must be 'text' or 'json'."
        try:
            if fmt == "json":
                tree_json = json.dumps(self._build_tree_json(resolved, depth=0, max_depth=max_depth), ensure_ascii=False)
                return self._truncate_output(tree_json)
            tree_text = self._build_tree_text(resolved, depth=0, max_depth=max_depth)
            return self._truncate_output(tree_text)
        except Exception as exc:
            return f"ERROR: failed to build tree for {resolved}: {exc}"

    def _build_tree_json(self, path: Path, depth: int, max_depth: int) -> Dict[str, Any]:
        node: Dict[str, Any] = {"name": path.name, "type": "dir" if path.is_dir() else "file"}
        if not path.is_dir() or depth >= max_depth:
            return node
        children: List[Dict[str, Any]] = []
        try:
            for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                children.append(self._build_tree_json(child, depth + 1, max_depth))
        except Exception as exc:
            node["error"] = f"failed to iterate directory: {exc}"
            return node
        node["children"] = children
        return node

    def _build_tree_text(self, path: Path, depth: int, max_depth: int, prefix: str = "") -> str:
        label = f"{path.name}/" if path.is_dir() else path.name
        lines = [label] if depth == 0 else []
        if not path.is_dir() or depth >= max_depth:
            return "\n".join(lines)

        try:
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception as exc:
            lines.append(f"{prefix}└── <error: failed to iterate directory: {exc}>")
            return "\n".join(lines)
        for i, child in enumerate(children):
            branch = "└── " if i == len(children) - 1 else "├── "
            child_label = f"{child.name}/" if child.is_dir() else child.name
            lines.append(f"{prefix}{branch}{child_label}")
            if child.is_dir() and depth + 1 < max_depth:
                extension = "    " if i == len(children) - 1 else "│   "
                subtree = self._build_tree_text(child, depth + 1, max_depth, prefix + extension)
                subtree_lines = subtree.splitlines()
                if subtree_lines:
                    lines.extend(subtree_lines[1:])
        return "\n".join(lines)

    def _register_extension(self, extension_path: Path) -> None:
        rel_path = str(extension_path.relative_to(self.root))
        now = datetime.now(timezone.utc).isoformat()
        entries: List[Dict[str, Any]] = []
        try:
            if self.extensions_registry_file.exists():
                loaded = json.loads(self.extensions_registry_file.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    entries = [e for e in loaded if isinstance(e, dict)]
        except Exception:
            entries = []

        updated = False
        for entry in entries:
            if entry.get("path") == rel_path:
                if "created_at" not in entry:
                    entry["created_at"] = now
                entry["updated_at"] = now
                updated = True
                break
        if not updated:
            entries.append({"path": rel_path, "created_at": now, "updated_at": now})
        self.extensions_registry_file.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def status(self) -> str:
        payload = {
            "phase": self.state.phase,
            "plan_emitted_this_cycle": self.state.plan_emitted_this_cycle,
            "snapshot_saved_this_cycle": self.state.snapshot_saved_this_cycle,
            "implementation_started": self.state.implementation_started,
            "shell_path": str(self.shell_dir),
            "snapshot_path": str(self.snapshot_dir),
            "extensions_registry_path": str(self.extensions_registry_file),
        }
        return json.dumps(payload, ensure_ascii=False)

    def _sanitize_snapshot_name(self, name: str) -> str:
        base = name.strip() or "snapshot"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", base)
        safe = safe.strip(".-_") or "snapshot"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{safe}-{stamp}"

    def _safe_extract_tar(self, archive: Path, destination: Path) -> None:
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                member_path = destination / member.name
                resolved = member_path.resolve()
                if not self._is_within(resolved, destination.resolve()):
                    raise RuntimeError(f"unsafe path in archive: {member.name}")
            tar.extractall(path=destination)

    def snapshot(self, mode: str, name: str) -> str:
        mode_clean = (mode or "").strip().lower()
        if mode_clean not in {"save", "restore"}:
            return "ERROR: snapshot mode must be 'save' or 'restore'."
        if not name or not name.strip():
            return "ERROR: snapshot name is required."

        if mode_clean == "save":
            if not self.state.plan_emitted_this_cycle:
                return "ERROR: snapshot save blocked. You must emit a planning response in this cycle before saving a snapshot."
            safe_name = self._sanitize_snapshot_name(name)
            archive = self.snapshot_dir / f"{safe_name}.tar.gz"
            try:
                with tarfile.open(archive, "w:gz") as tar:
                    tar.add(self.shell_dir, arcname="shell")
                self.state.mark_snapshot_saved()
                return f"OK: snapshot saved to {archive.name}"
            except Exception as exc:
                return f"ERROR: snapshot save failed: {exc}"

        requested = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
        candidate = self.snapshot_dir / f"{requested}.tar.gz"
        if not candidate.exists():
            matches = sorted(self.snapshot_dir.glob(f"{requested}*.tar.gz"))
            if len(matches) == 1:
                candidate = matches[0]
            else:
                return f"ERROR: snapshot not found for name '{name}'."
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="pi-restore-", dir=str(self.root)))
            try:
                self._safe_extract_tar(candidate, tmp_dir)
                restored_shell = tmp_dir / "shell"
                if not restored_shell.exists() or not restored_shell.is_dir():
                    return "ERROR: invalid snapshot format (missing 'shell/' root)."

                if self.shell_dir.exists():
                    shutil.rmtree(self.shell_dir)
                shutil.copytree(restored_shell, self.shell_dir)
                self.extensions_dir.mkdir(parents=True, exist_ok=True)
                self.state.phase = "plan"
                self.state.plan_emitted_this_cycle = False
                self.state.snapshot_saved_this_cycle = False
                self.state.implementation_started = False
                return f"OK: snapshot restored from {candidate.name}"
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:
            return f"ERROR: snapshot restore failed: {exc}"

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read UTF-8 text from a file path inside the project root.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "description": "Overwrite a file with text content (default policy allows writes under .pi/shell/).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit",
                    "description": "Literal search-and-replace. Replaces all occurrences of old_str.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_str": {"type": "string"},
                            "new_str": {"type": "string"},
                        },
                        "required": ["path", "old_str", "new_str"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a shell command in project root and return JSON with stdout/stderr/exit_code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                        },
                        "required": ["cmd"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_snapshots",
                    "description": "List available snapshots in .pi/core/snapshots/ as JSON.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "status",
                    "description": "Return current session workflow state and key workspace paths as JSON.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "snapshot",
                    "description": "Save/restore .pi/shell snapshots in .pi/core/snapshots as tar.gz.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {"type": "string", "enum": ["save", "restore"]},
                            "name": {"type": "string"},
                        },
                        "required": ["mode", "name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tree",
                    "description": "Inspect the workspace tree. Defaults to .pi/shell/ and depth 3.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "default": SHELL_DIR},
                            "max_depth": {"type": "integer", "default": 3},
                            "format": {"type": "string", "enum": ["text", "json"], "default": "text"},
                        },
                    },
                },
            },
        ]


def parse_tool_args(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def call_model(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    response = ollama.chat(model=MODEL_NAME, messages=messages, tools=tools)
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return json.loads(json.dumps(response, default=lambda x: getattr(x, "__dict__", str(x))))


def extract_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = get_field(message, "tool_calls", []) or []
    normalized: List[Dict[str, Any]] = []
    for c in calls:
        fn = get_field(c, "function", {})
        normalized.append(
            {
                "id": get_field(c, "id", ""),
                "name": get_field(fn, "name", ""),
                "arguments": get_field(fn, "arguments", {}),
            }
        )
    return normalized


def tool_dispatch(agent: PiAgent) -> Dict[str, Callable[..., str]]:
    return {
        "read": lambda **kw: agent.read(path=kw.get("path", "")),
        "write": lambda **kw: agent.write(path=kw.get("path", ""), content=kw.get("content", "")),
        "edit": lambda **kw: agent.edit(
            path=kw.get("path", ""), old_str=kw.get("old_str", ""), new_str=kw.get("new_str", "")
        ),
        "bash": lambda **kw: agent.bash(cmd=kw.get("cmd", "")),
        "list_snapshots": lambda **kw: agent.list_snapshots(),
        "status": lambda **kw: agent.status(),
        "snapshot": lambda **kw: agent.snapshot(mode=kw.get("mode", ""), name=kw.get("name", "")),
        "tree": lambda **kw: agent.tree(
            path=kw.get("path", SHELL_DIR),
            max_depth=kw.get("max_depth", 3),
            format=kw.get("format", "text"),
        ),
    }


def run_assistant_turn(agent: PiAgent, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    tools = agent.get_tools()
    dispatch = tool_dispatch(agent)

    while True:
        resp = call_model(messages, tools)
        msg = get_field(resp, "message", {})
        role = get_field(msg, "role", "assistant")
        content = get_field(msg, "content", "")
        tool_calls = extract_tool_calls(msg)

        assistant_msg: Dict[str, Any] = {"role": role}
        if content:
            assistant_msg["content"] = content
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for c in tool_calls
            ]
        messages.append(assistant_msg)

        if not tool_calls:
            if content:
                if content.strip().startswith("Plan:"):
                    agent.state.mark_plan_emitted()
                agent.state.mark_verify()
            return messages, content

        for c in tool_calls:
            tool_name = c["name"]
            args = parse_tool_args(c.get("arguments", {}))
            if tool_name not in dispatch:
                result = f"ERROR: unknown tool '{tool_name}'"
            else:
                try:
                    result = dispatch[tool_name](**args)
                except Exception as exc:
                    result = f"ERROR: tool '{tool_name}' failed: {exc}"

            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": result,
                }
            )


def main() -> int:
    root = Path.cwd()
    agent = PiAgent(root)
    agent.init_layout()
    agents_md = agent.read_agents_md()

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": f"Current AGENTS.md policy:\n\n{agents_md}",
        },
    ]

    messages.append(
        {
            "role": "user",
            "content": "Start the session. Introduce yourself briefly and ask exactly one opening question to begin planning.",
        }
    )

    try:
        messages, opening = run_assistant_turn(agent, messages)
        if opening:
            print(f"assistant> {opening}")
        else:
            print("assistant> Hello. What would you like to build?")

        while True:
            try:
                user_input = input("you> ")
            except EOFError:
                print("\nExiting.")
                break

            if not user_input.strip():
                continue

            lowered = user_input.strip().lower()
            if lowered in {"exit", "quit"}:
                print("Exiting.")
                break

            agent.state.begin_new_task_block()
            messages.append({"role": "user", "content": user_input})
            messages, final_content = run_assistant_turn(agent, messages)
            if final_content:
                print(f"assistant> {final_content}")
            else:
                print("assistant> (no response)")

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
