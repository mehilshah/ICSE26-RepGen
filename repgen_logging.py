"""
Shared logging utilities for RepGen.

This module standardizes console and file logs into an agent-style trace with
consistent context fields across the retrieval and generation pipeline.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")


def _success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)


if not hasattr(logging.Logger, "success"):
    logging.Logger.success = _success  # type: ignore[attr-defined]


class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that merges bound context with per-call extras."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        extra = dict(self.extra)
        extra.update(kwargs.get("extra", {}))
        kwargs["extra"] = extra
        return msg, kwargs

    def bind(self, **extra: Any) -> "ContextAdapter":
        merged = dict(self.extra)
        merged.update(extra)
        return ContextAdapter(self.logger, merged)

    def success(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.log(SUCCESS_LEVEL_NUM, msg, *args, **kwargs)


def get_logger(name: Optional[str] = None, **context: Any) -> ContextAdapter:
    return ContextAdapter(logging.getLogger(name), context)


class AgentTraceFormatter(logging.Formatter):
    RESET = "\x1b[0m"
    GREY = "\x1b[38;5;240m"
    BLUE = "\x1b[38;5;39m"
    GREEN = "\x1b[38;5;82m"
    YELLOW = "\x1b[38;5;226m"
    RED = "\x1b[38;5;196m"
    BOLD_RED = "\x1b[31;1m"
    CYAN = "\x1b[38;5;51m"
    MAGENTA = "\x1b[38;5;213m"

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def _color(self, value: str, color: str) -> str:
        if not self.use_color:
            return value
        return f"{color}{value}{self.RESET}"

    def _format_level(self, levelno: int) -> str:
        mapping = {
            logging.DEBUG: self._color("DEBUG", self.GREY),
            logging.INFO: self._color("INFO", self.BLUE),
            SUCCESS_LEVEL_NUM: self._color("SUCCESS", self.GREEN),
            logging.WARNING: self._color("WARNING", self.YELLOW),
            logging.ERROR: self._color("ERROR", self.RED),
            logging.CRITICAL: self._color("CRITICAL", self.BOLD_RED),
        }
        return mapping.get(levelno, self._color("LOG", self.CYAN))

    def _tag(self, label: str, value: Any, color: str) -> str:
        return self._color(f"{label}={value}", color)

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self._color(self.formatTime(record, "%H:%M:%S"), self.GREY)
        level = self._format_level(record.levelno)

        component = getattr(record, "component", record.name)
        tags = [self._tag("component", component, self.CYAN)]

        for label, key, color in (
            ("bug", "bug_id", self.MAGENTA),
            ("ctx", "context_num", self.CYAN),
            ("try", "attempt", self.YELLOW),
            ("stage", "stage", self.BLUE),
            ("action", "action", self.GREEN),
            ("status", "status", self.MAGENTA),
            ("model", "model", self.CYAN),
            ("backend", "backend", self.YELLOW),
        ):
            value = getattr(record, key, None)
            if value is not None:
                tags.append(self._tag(label, value, color))

        message = record.getMessage()
        details = getattr(record, "details", None)
        if details:
            if isinstance(details, dict):
                suffix = ", ".join(f"{key}={value}" for key, value in details.items())
            else:
                suffix = str(details)
            message = f"{message} | {suffix}"

        return f"{timestamp} [{level}] {' '.join(tags)} {message}"


def setup_logging(log_level: int = logging.INFO, log_file: Optional[str] = None) -> ContextAdapter:
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(AgentTraceFormatter(use_color=True))
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(AgentTraceFormatter(use_color=False))
        root_logger.addHandler(file_handler)

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    logger = get_logger("repgen", component="repgen")
    if log_file:
        trace(logger, "File logging enabled", action="configure_logging", status="ok", details={"log_file": log_file})
    return logger


def trace(
    logger: ContextAdapter,
    message: str,
    *,
    level: int = logging.INFO,
    details: Optional[Dict[str, Any]] = None,
    **extra: Any,
) -> None:
    payload = dict(extra)
    if details:
        payload["details"] = details
    logger.log(level, message, extra=payload)

