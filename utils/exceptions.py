"""Custom exception hierarchy for the project."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Type

__all__ = [
    "AIError",
    "ConfigurationError",
    "DependencyError",
    "ValidationError",
    "ExternalServiceError",
    "PipelineStepError",
    "UserAbort",
    "RetryableError",
    "ensure",
    "exception_to_dict",
]


class AIError(Exception):
    """Base exception for the project."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.message = message or self.__class__.__name__
        self.details: dict[str, Any] = dict(details or {})
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(
                f"{key}={value}" for key, value in self.details.items()
            )
            return f"{self.message} ({detail_str})"
        return self.message


class ConfigurationError(AIError):
    """Raised when configuration values are missing or invalid."""


class DependencyError(AIError):
    """Raised when required dependencies are unavailable or misconfigured."""


class ValidationError(AIError):
    """Raised when user-provided inputs fail validation."""


class ExternalServiceError(AIError):
    """Raised when communication with an external service fails."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        service: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        merged_details = dict(details or {})
        if service is not None:
            merged_details.setdefault("service", service)
        if status_code is not None:
            merged_details.setdefault("status_code", status_code)

        base_message = message or "External service request failed"
        if service and message is None:
            base_message = f"External service '{service}' request failed"

        self.service = service
        self.status_code = status_code

        super().__init__(base_message, details=merged_details)


class RetryableError(AIError):
    """Raised for errors that may succeed when retried."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        attempts: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        merged_details = dict(details or {})
        if attempts is not None:
            merged_details.setdefault("attempts", attempts)

        base_message = message or "Retryable operation failed"
        super().__init__(base_message, details=merged_details)
        self.attempts = attempts


class PipelineStepError(AIError):
    """Raised when a specific pipeline step fails."""

    def __init__(
        self,
        step: str,
        message: Optional[str] = None,
        *,
        original_exc: Optional[BaseException] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        merged_details = dict(details or {})
        merged_details.setdefault("step", step)
        if original_exc is not None:
            merged_details.setdefault("cause", repr(original_exc))

        base_message = message or f"Pipeline step '{step}' failed"
        if original_exc is not None and message is None:
            base_message = f"{base_message}: {original_exc}"

        self.step = step
        self.original_exc = original_exc

        super().__init__(base_message, details=merged_details)


class UserAbort(AIError):
    """Raised when execution is intentionally halted by the user."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        base_message = message or "Operation aborted by user request"
        super().__init__(base_message, details=details)


def ensure(
    condition: bool,
    message: str,
    *,
    exception_cls: Type[AIError] = ValidationError,
    **exception_kwargs: Any,
) -> None:
    """Assert a condition and raise the provided exception if it fails."""
    if not condition:
        raise exception_cls(message, **exception_kwargs)


def exception_to_dict(exc: BaseException) -> dict[str, Any]:
    """Represent an exception as a serialisable dictionary."""
    payload: dict[str, Any] = {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }
    if isinstance(exc, AIError):
        payload["details"] = dict(exc.details)
    return payload
