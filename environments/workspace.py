"""Code workspace integration: a sandboxed folder rendered as a screen (listing, open file, results).

Writes and commands never leave the workspace root. Commands are limited to an allow-list and a timeout.
Network isolation is the host's job (run inside a sandbox or container).
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from body.schema import Observation, Screen, ScreenSpec, SensoryManifest

from .base import Integration

WIDTH, HEIGHT, ROW = 1024, 768, 16
ALLOWED = ("python", "python3", "pytest")


class WorkspaceEnvironment(Integration):
    def __init__(self, root: str | Path, goal: str = "", allowed: tuple[str, ...] = ALLOWED, timeout_s: int = 60):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._goal, self.allowed, self.timeout_s = goal, tuple(allowed), int(timeout_s)
        self.reset()

    # ------------------------------------------------------------------ lifecycle and sensing
    def reset(self, seed: Optional[int] = None) -> None:
        self.cwd = self.root
        self.open_path: Optional[Path] = None
        self.hits: list[tuple[str, int, str]] = []
        self.output = ""
        self.layout: dict = {}

    @property
    def goal(self) -> str:
        return self._goal

    def sensory_manifest(self) -> SensoryManifest:
        return SensoryManifest(screens=[ScreenSpec("flat", WIDTH, HEIGHT)], realtime=False)

    def observe(self) -> Observation:
        return Observation(screens=[Screen("flat", self.render())], pointer=None)

    # ------------------------------------------------------------------ paths
    def inside(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if path != self.root and self.root not in path.parents:
            raise PermissionError(f"{relative} is outside the workspace")
        return path

    def rel(self, path: Path) -> str:
        return str(path.relative_to(self.root)).replace("\\", "/") or "."

    def entries(self, directory: Optional[Path] = None) -> list[Path]:
        directory = directory or self.cwd
        return sorted(p for p in directory.iterdir() if not p.name.startswith("."))

    # ------------------------------------------------------------------ operations
    def open(self, relative: str) -> str:
        path = self.inside(relative)
        if path.is_dir():
            self.cwd = path
            return f"opened folder {self.rel(path)}"
        self.open_path = path
        return f"opened {self.rel(path)}"

    def read(self, relative: str, limit: int = 4000) -> str:
        return self.inside(relative).read_text(encoding="utf-8", errors="replace")[:limit]

    def search(self, query: str, limit: int = 30) -> str:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        self.hits = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg"}:
                for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if pattern.search(line):
                        self.hits.append((self.rel(path), number, line.strip()[:120]))
                        if len(self.hits) >= limit:
                            return f"{len(self.hits)} matches (truncated)"
        return f"{len(self.hits)} matches"

    def write(self, relative: str, content: str) -> str:
        path = self.inside(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.open_path = path
        return f"wrote {self.rel(path)} ({len(content)} chars)"

    def run(self, command: str) -> tuple[bool, str]:
        args = shlex.split(command)
        if not args or args[0] not in self.allowed:
            return False, f"command not allowed: {args[0] if args else '(empty)'}"
        try:
            done = subprocess.run(args, cwd=self.root, capture_output=True, text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            return False, "timed out"
        self.output = (done.stdout + done.stderr)[-4000:]
        return done.returncode == 0, f"exit {done.returncode}"

    # ------------------------------------------------------------------ screen
    def render(self) -> np.ndarray:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (WIDTH, HEIGHT), "white")
        draw = ImageDraw.Draw(image)
        self.layout = {}
        y = 4
        draw.text((8, y), f"[{self.rel(self.cwd)}]", fill="black")
        for path in self.entries()[:40]:
            y += ROW
            key = ("dir" if path.is_dir() else "file", self.rel(path))
            self.layout[key] = (0.0, y / HEIGHT, 0.3, (y + ROW) / HEIGHT)
            draw.text((16, y), path.name + ("/" if path.is_dir() else ""), fill="black")
        if self.open_path is not None and self.open_path.exists():
            draw.text((320, 4), self.rel(self.open_path), fill="black")
            self.layout[("open", self.rel(self.open_path))] = (0.31, 0.0, 1.0, 0.6)
            for i, line in enumerate(self.open_path.read_text(encoding="utf-8", errors="replace").splitlines()[:28]):
                draw.text((320, 4 + (i + 1) * ROW), line[:90], fill="black")
        y = int(HEIGHT * 0.62)
        for i, (path, number, line) in enumerate(self.hits[:8]):
            self.layout[("hit", path, number)] = (0.0, (y + i * ROW) / HEIGHT, 1.0, (y + (i + 1) * ROW) / HEIGHT)
            draw.text((8, y + i * ROW), f"{path}:{number}: {line}", fill="black")
        self.layout[("terminal",)] = (0.0, 0.9, 1.0, 1.0)
        draw.text((8, int(HEIGHT * 0.9)), "$ " + (self.output.splitlines()[-1] if self.output else ""), fill="black")
        return np.asarray(image, dtype=np.uint8)
