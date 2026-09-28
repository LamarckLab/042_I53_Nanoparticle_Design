"""Backend abstraction: run a tool inside a conda env, a container, or the current shell.

Stages build a plain argv list and hand it to a Runner. Swapping conda for Docker at
publication time is then a config change, not a code change.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class Runner:
    kind: str = "local"                      # local | conda | docker | apptainer
    envs: dict = field(default_factory=dict)
    images: dict = field(default_factory=dict)
    gpus: list = field(default_factory=lambda: [0])
    cuda_device_order: str = "PCI_BUS_ID"
    log_path: Path | None = None
    dry_run: bool = False
    binds: list = field(default_factory=list)

    # -- environment --------------------------------------------------------
    def env_vars(self) -> dict:
        env = dict(os.environ)
        # PCI_BUS_ID first: without it CUDA reorders devices by capability and
        # CUDA_VISIBLE_DEVICES selects a different card than the one intended.
        env["CUDA_DEVICE_ORDER"] = self.cuda_device_order
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpus)
        return env

    # -- command construction ----------------------------------------------
    def wrap(self, tool: str, argv: list[str], workdir: Path | None = None) -> list[str]:
        if self.kind == "local":
            return list(argv)
        if self.kind == "conda":
            env = self.envs.get(tool)
            if not env:
                raise KeyError(f"no conda env configured for tool {tool!r}")
            return ["conda", "run", "-n", env, "--no-capture-output", *argv]
        if self.kind in ("docker", "apptainer"):
            image = self.images.get(tool)
            if not image:
                raise KeyError(f"no {self.kind} image configured for tool {tool!r}")
            mounts: list[str] = []
            for src in self.binds:
                mounts += ["-v", f"{src}:{src}"] if self.kind == "docker" else ["--bind", f"{src}:{src}"]
            if self.kind == "docker":
                gpu_flag = ["--gpus", f'"device={",".join(str(g) for g in self.gpus)}"'] if self.gpus else []
                wd = ["-w", str(workdir)] if workdir else []
                return ["docker", "run", "--rm", *gpu_flag, *mounts, *wd, image, *argv]
            wd = ["--pwd", str(workdir)] if workdir else []
            return ["apptainer", "exec", "--nv", *mounts, *wd, image, *argv]
        raise ValueError(f"unknown backend kind: {self.kind!r}")

    # -- execution ----------------------------------------------------------
    def run(self, tool: str, argv: list[str], workdir: Path | None = None,
            check: bool = True) -> CommandResult:
        full = self.wrap(tool, argv, workdir)
        printable = " ".join(shlex.quote(a) for a in full)

        if self.dry_run:
            self._log({"tool": tool, "dry_run": True, "cmd": printable})
            return CommandResult(full, 0, "", "", 0.0)

        start = time.time()
        proc = subprocess.run(
            full, cwd=str(workdir) if workdir else None, env=self.env_vars(),
            capture_output=True, text=True,
        )
        result = CommandResult(full, proc.returncode, proc.stdout, proc.stderr, time.time() - start)
        self._log({
            "tool": tool, "cmd": printable, "returncode": result.returncode,
            "seconds": round(result.seconds, 2),
            "stderr_tail": result.stderr[-2000:] if result.stderr else "",
        })
        if check and not result.ok:
            raise RuntimeError(
                f"{tool} failed (exit {result.returncode})\n"
                f"  cmd: {printable}\n"
                f"  stderr: {result.stderr[-2000:]}"
            )
        return result

    def _log(self, record: dict) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def runner_from_config(cfg, outdir: Path, dry_run: bool = False) -> Runner:
    return Runner(
        kind=str(cfg.get("backend.kind", "local")),
        envs=cfg.get("backend.envs", {}) or {},
        images=cfg.get("backend.images", {}) or {},
        gpus=cfg.get("compute.gpus", [0]) or [0],
        cuda_device_order=str(cfg.get("compute.cuda_device_order", "PCI_BUS_ID")),
        log_path=Path(outdir) / "logs" / "commands.jsonl",
        dry_run=dry_run,
        binds=[str(outdir)],
    )
