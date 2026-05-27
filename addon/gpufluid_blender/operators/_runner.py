"""[Round-14] ModalSubprocessRunner — shared lifecycle for OT_bake + OT_render.

Senior architect (round-12) day-1 rewrite #2. Both operators carried
identical subprocess plumbing — Popen + background drain thread + Queue
+ modal timer + ESC abort + cancel() callback + sync mode + watchdog
+ OSError guard. Rounds 1-12 fixed each operator twice: once when the
bug was found in one, once again later when the mirror operator drifted
(lesson 9.6 — mirror-operator drift). This base kills that duplication
permanently.

**Reentrance flag (`_is_running`) stays on the operator class** so
Blender's "one instance per click" model still works as the natural
mutex. The runner just reads/writes it via `operator.__class__`.

Contract (encoded by these methods + invariant tests):

  start_sync(op, argv, timeout, log_prefix, on_complete)
    Spawn subprocess + drain stdout on a daemon thread + wait(timeout).
    On rc=0 invoke on_complete callback (auto-attach hook). On timeout,
    kill+wait+report. On OSError during spawn-or-thread-setup, clear
    flag + report. ALWAYS clears `op.__class__._is_running` before
    returning.

  start_modal(op, context, argv, status_msg)
    Spawn + drain via subprocess_drain helper + arm event_timer +
    modal_handler_add. Sets class flag. Returns RUNNING_MODAL or
    CANCELLED on Popen/Queue/Thread failure.

  tick_modal(op, context, event, frame_regex, on_progress)
    Handle ESC (→ abort) and TIMER (→ poll + drain queue + check rc).
    Returns CANCELLED/FINISHED set when done, None to PASS_THROUGH.

  abort(op, context, reason)
    Terminate(timeout=3) → kill(timeout=2). Remove timer, clear status,
    clear instance attrs. Always returns CANCELLED.

  cancel(op, context)
    Routes to abort. Blender's cancel() callback (window-close etc.).

This file is bpy-aware (imports `bpy` lazily in methods that need it
so unit tests can import the class without a live Blender).
"""
from __future__ import annotations

import queue
import re
import subprocess
import threading
from typing import Callable, Optional

from .helpers import subprocess_drain


class ModalSubprocessRunner:
    """Per-operator-instance lifecycle holder. One runner per Operator
    instance (lazy-init in execute()); the operator class itself holds
    the reentrance flag for cross-instance mutex semantics."""

    def __init__(self, label: str) -> None:
        """`label` is "bake" / "render" — used in log messages, status
        bar text, and report() strings."""
        self.label = label
        # Subprocess lifecycle state — all None outside a live run.
        self._proc: Optional[subprocess.Popen] = None
        self._stdout_q: Optional[queue.Queue] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._timer = None
        # For modal status-bar updates
        self._current_frame: int = 0
        self._total_frames: int = 0

    # ── Internal helpers ────────────────────────────────────────────

    def _clear_instance_state(self) -> None:
        """Drop refs to proc/queue/thread/timer. Called from abort/finish
        so a re-fired modal on the same instance after cancel doesn't
        see stale state and short-circuit straight to _finish(ok=False).
        Round-7 + round-9 reviewers caught the bake + render asymmetric
        cleanup; runner encodes it once."""
        self._proc = None
        self._stdout_q = None
        self._stdout_thread = None
        self._timer = None

    def _spawn_and_drain(self, argv: list[str]) -> None:
        """Popen + Queue + drain thread. Raises (OSError, RuntimeError)
        on failure with the orphan-Popen-kill cleanup already done."""
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, encoding="utf-8", errors="replace",
        )
        try:
            self._stdout_q = queue.Queue()
            self._stdout_thread = threading.Thread(
                target=subprocess_drain,
                args=(self._proc, self._stdout_q),
                daemon=True,
            )
            self._stdout_thread.start()
        except Exception:
            # Round-9 reviewer: if Queue/Thread setup fails after Popen
            # success, kill the now-orphan subprocess so it doesn't run
            # without a drain.
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.kill(); self._proc.wait(timeout=2.0)
                except Exception:
                    pass
            raise

    # ── Public lifecycle: sync mode ─────────────────────────────────

    def start_sync(
        self,
        operator,
        argv: list[str],
        timeout_sec: int,
        log_prefix: str,
        on_complete: Optional[Callable[[], None]] = None,
        logger=None,
        friendly_error_for_rc: Optional[Callable[[int], Optional[str]]] = None,
    ) -> set[str]:
        """Synchronous run: spawn + drain on bg thread + wait(timeout).
        Returns operator-result set ({'FINISHED'} or {'CANCELLED'}).
        Logs each stdout line via `logger.info(f"{log_prefix}: %s", line)`
        if a logger is provided, else swallows.

        Lifecycle:
          1. Set `op.__class__._is_running = True` (caller-side
             responsibility BEFORE calling this — checked at top).
          2. Popen + drain-thread (round-9 OSError-around-all guard).
          3. wait(timeout=timeout_sec or None).
             - TimeoutExpired → kill + wait + flag-clear + ERROR + return.
          4. join drain thread → log accumulated lines.
          5. rc check → ERROR if non-zero.
          6. on_complete() callback (auto-attach).
          7. INFO report + FINISHED.

        `op.__class__._is_running` ALWAYS cleared before return via
        try/finally — even on OSError between sub-paths.
        """
        op_class = operator.__class__
        timeout = int(timeout_sec) if timeout_sec > 0 else None
        drained: list[str] = []

        def _drain_lines(p, out):
            try:
                for line in p.stdout:
                    out.append(line)
            except Exception:
                pass

        try:
            try:
                self._proc = subprocess.Popen(
                    argv,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    bufsize=1, text=True, encoding="utf-8", errors="replace",
                )
            except OSError as e:
                operator.report(
                    {"ERROR"},
                    f"could not spawn {self.label} subprocess: {e}")
                return {"CANCELLED"}

            # Round-23: mirror the orphan-Popen-kill guard the modal
            # path got in round-9 (`_spawn_and_drain`). If Thread()
            # construction or start() fails (rare: resource exhaustion),
            # the subprocess we just spawned runs without a drainer
            # and its handle leaks. Same defence applied here so the
            # sync/modal paths stay symmetric (lesson 9.6 — mirror-
            # operator drift inside a single class).
            try:
                drain_thread = threading.Thread(
                    target=_drain_lines, args=(self._proc, drained),
                    daemon=True)
                drain_thread.start()
            except Exception as e:
                if self._proc is not None and self._proc.poll() is None:
                    try:
                        self._proc.kill(); self._proc.wait(timeout=2.0)
                    except Exception:
                        pass
                operator.report(
                    {"ERROR"},
                    f"could not start {self.label} drainer thread: {e}")
                return {"CANCELLED"}
            try:
                rc = self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5.0)
                operator.report(
                    {"ERROR"},
                    f"gpufluid {self.label} (sync) hit timeout after "
                    f"{timeout}s — subprocess killed.")
                return {"CANCELLED"}
            drain_thread.join(timeout=1.0)
            if logger is not None:
                for line in drained:
                    logger.info("%s: %s", log_prefix, line.rstrip())
            if rc != 0:
                # Round-20: callers (notably OT_bake for MPM) may translate
                # specific non-zero rcs into actionable error messages by
                # reading sidecars like cache.json. Falls through to the
                # generic "rc=N" wording when callback returns None or absent.
                friendly = None
                if friendly_error_for_rc is not None:
                    try:
                        friendly = friendly_error_for_rc(rc)
                    except Exception:
                        friendly = None
                operator.report(
                    {"ERROR"},
                    friendly or
                    f"gpufluid {self.label} (sync) failed with rc={rc}")
                return {"CANCELLED"}
            if on_complete is not None:
                try:
                    on_complete()
                except Exception as e:
                    operator.report(
                        {"WARNING"},
                        f"sync {self.label} ok, but on_complete failed: {e}")
            operator.report(
                {"INFO"}, f"gpufluid {self.label} (sync) finished")
            return {"FINISHED"}
        finally:
            # Caller set the flag; we always clear on return so the
            # next invocation doesn't see a stale lock.
            op_class._is_running = False
            self._clear_instance_state()

    # ── Public lifecycle: modal mode ────────────────────────────────

    def start_modal(
        self,
        operator,
        context,
        argv: list[str],
        status_msg: str,
    ) -> set[str]:
        """Spawn subprocess + drain thread (via subprocess_drain helper,
        which uses a sentinel-terminated Queue protocol so the modal
        timer can drain non-blockingly) + arm timer + register modal.

        Returns RUNNING_MODAL on success or CANCELLED on spawn failure.
        On failure, clears the class flag the caller set."""
        import bpy
        op_class = operator.__class__
        try:
            self._spawn_and_drain(argv)
        except (OSError, RuntimeError) as e:
            self._clear_instance_state()
            op_class._is_running = False
            operator.report(
                {"ERROR"},
                f"could not spawn {self.label} subprocess: {e}")
            return {"CANCELLED"}

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.3, window=context.window)
        wm.modal_handler_add(operator)
        context.workspace.status_text_set(status_msg)
        return {"RUNNING_MODAL"}

    def tick_modal(
        self,
        operator,
        context,
        event,
        frame_regex: Optional[re.Pattern] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        logger=None,
        log_prefix: Optional[str] = None,
        friendly_error_for_rc: Optional[Callable[[int], Optional[str]]] = None,
    ) -> Optional[set[str]]:
        """Called from operator.modal(). Handles ESC + TIMER + drain.

        Returns:
          - {'CANCELLED'} on ESC (after abort)
          - {'CANCELLED'} / {'FINISHED'} when subprocess exits
          - None if event was PASS_THROUGH (caller returns it)
        """
        if event.type == "ESC":
            return self.abort(operator, context, reason="user pressed Esc")
        if event.type != "TIMER":
            return None  # caller returns {"PASS_THROUGH"}
        if self._proc is None:
            return self._finish(operator, context, ok=False)

        rc = self._proc.poll()
        if self._stdout_q is not None:
            for _ in range(50):
                try:
                    line = self._stdout_q.get_nowait()
                except queue.Empty:
                    break
                if line is None:   # sentinel: stdout EOF
                    break
                if frame_regex is not None:
                    m = frame_regex.search(line)
                    if m:
                        self._current_frame = int(m.group(1))
                        self._total_frames = int(m.group(2))
                        if on_progress is not None:
                            on_progress(self._current_frame, self._total_frames)
                if logger is not None and log_prefix is not None:
                    logger.info("%s: %s", log_prefix, line.rstrip())
        if rc is None:
            return None  # caller returns {"PASS_THROUGH"}
        return self._finish(operator, context, ok=(rc == 0),
                            rc=rc,
                            friendly_error_for_rc=friendly_error_for_rc)

    def abort(self, operator, context, reason: str) -> set[str]:
        """Terminate(timeout=3) → kill(timeout=2). Idempotent — safe to
        call from both modal(ESC) and Blender's cancel() callback.
        Always returns CANCELLED."""
        op_class = operator.__class__
        op_class._is_running = False
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
            except Exception:
                pass
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
        context.workspace.status_text_set(None)
        self._clear_instance_state()
        operator.report({"WARNING"},
                        f"gpufluid {self.label} aborted ({reason})")
        return {"CANCELLED"}

    def cancel(self, operator, context) -> None:
        """Blender callback for right-click/window-close. Route through
        abort so subprocess teardown always happens."""
        self.abort(operator, context, reason="cancelled by Blender")

    # ── _finish helper (used by tick_modal end-of-subprocess) ───────

    def _finish(
        self,
        operator,
        context,
        ok: bool,
        rc: Optional[int] = None,
        friendly_error_for_rc: Optional[Callable[[int], Optional[str]]] = None,
    ) -> set[str]:
        """End-of-modal cleanup. `on_done` (auto-attach) called by the
        operator AFTER this if needed — runner doesn't know operator-
        specific success behaviour. Returns {'FINISHED'} or {'CANCELLED'}.

        Round-20: ``friendly_error_for_rc`` (optional) lets the operator
        translate a specific non-zero rc into an actionable message
        (e.g. OT_bake reads ``cache.json`` to surface MPM divergence
        frame on rc=2). Returns None → generic fallback wording."""
        op_class = operator.__class__
        op_class._is_running = False
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
        context.workspace.status_text_set(None)
        self._clear_instance_state()
        if not ok:
            friendly = None
            if rc is not None and friendly_error_for_rc is not None:
                try:
                    friendly = friendly_error_for_rc(rc)
                except Exception:
                    friendly = None
            operator.report(
                {"ERROR"},
                friendly or
                f"gpufluid {self.label} failed — see system console")
            return {"CANCELLED"}
        # Caller is responsible for INFO report + any auto-attach.
        return {"FINISHED"}
