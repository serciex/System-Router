"""Code workspace starter adapter: files, folders, search hits and a terminal as items.

`write` takes JSON `{"path": "...", "content": "..."}`; `search` takes a query; `run` takes a command.
Required for adapter writing.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from ..schema import InteractionManifest, Item
from .base import Adapter, result


class CodeWorkspaceAdapter(Adapter):
    name = "code"

    def __init__(self, workspace):
        self.ws = workspace

    def manifest(self) -> InteractionManifest:
        return InteractionManifest(pointer_mode="absolute", verbs=("open", "read", "search", "write", "run"),
                                   self_verbs=("search", "write"), actuators={"keyboard": {}})

    def find(self, scope: Any = None) -> list[Item]:
        self.ws.render()
        layout = self.ws.layout
        if scope is not None and scope[0] == "dir":
            self.ws.open(scope[1])
            self.ws.render()
            layout = self.ws.layout
        items: list[Item] = []
        for key, box in layout.items():
            kind = key[0]
            if kind == "dir":
                count = len(self.ws.entries(self.ws.inside(key[1])))
                items.append(Item(handle=key, kind="group", role="group", name=key[1] + "/", verbs=(), bbox=box,
                                  collapsed=count))
            elif kind == "file":
                items.append(Item(handle=key, kind="element", role="file", name=key[1], verbs=("open", "read"), bbox=box))
            elif kind == "open":
                items.append(Item(handle=key, kind="element", role="file", name=f"{key[1]} (open)",
                                  verbs=("read", "write"), bbox=box))
            elif kind == "hit":
                items.append(Item(handle=key, kind="element", role="symbol", name=f"{key[1]}:{key[2]}",
                                  verbs=("open",), bbox=box))
            elif kind == "terminal":
                items.append(Item(handle=key, kind="element", role="textbox", name="terminal", verbs=("run",), bbox=box))
        return items

    def invoke(self, handle, verb: str, arg: Optional[str] = None) -> dict:
        try:
            if verb == "search":
                return result(True, self.ws.search(arg or ""))
            if verb == "write":
                spec = json.loads(arg or "{}")
                path = spec.get("path") or (handle[1] if handle and handle[0] == "open" else None)
                if not path:
                    return result(False, "write needs a path")
                return result(True, self.ws.write(path, spec.get("content", "")))
            if handle is None:
                return result(False, f"{verb} needs a target")
            if verb == "open":
                return result(True, self.ws.open(handle[1]))
            if verb == "read":
                return result(True, self.ws.read(handle[1]))
            if verb == "run":
                ok, text = self.ws.run(arg or "")
                return result(ok, text)
        except (PermissionError, FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            return result(False, str(error))
        return result(False, f"unsupported verb {verb}")
