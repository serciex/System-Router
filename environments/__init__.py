"""Environment integrations (sensory channel and episode lifecycle). Sandboxed only."""


def make_environment(cfg):
    suite = cfg.env.suite
    if suite == "miniwob":
        from .miniwob import MiniWoBEnvironment

        return MiniWoBEnvironment(tasks=list(cfg.env.tasks), render=bool(cfg.env.render))
    if suite == "workspace":
        from common.config import resolve

        from .workspace import WorkspaceEnvironment

        return WorkspaceEnvironment(resolve(cfg.workspace.root), goal=cfg.workspace.goal)
    if suite == "desktop":
        from .desktop import DesktopEnvironment

        return DesktopEnvironment()
    raise NotImplementedError(f"Environment suite {suite!r} is not implemented yet")
