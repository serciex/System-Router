"""Sandboxed environments. Only MiniWoB++ for now; WebArena and OSWorld come in stages 4-5.

Named `environments` (not `envs`) so it never shadows r2dreamer's own `envs` package.
"""


def make_environment(cfg):
    suite = cfg.env.suite
    if suite == "miniwob":
        from .miniwob import MiniWoBEnvironment

        return MiniWoBEnvironment(tasks=list(cfg.env.tasks), render=bool(cfg.env.render))
    raise NotImplementedError(f"Environment suite {suite!r} is not implemented yet")
