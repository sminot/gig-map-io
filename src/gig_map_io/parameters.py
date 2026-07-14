from __future__ import annotations

"""
Simple key-value parameter store backed by a JSON file.

The ``Parameters`` object maintains an in-memory mapping of keys to values,
and keeps it synchronized with a JSON file on disk. It is designed to store
simple, JSON-serializable keyword arguments.

When used in marimo notebooks, helper methods (e.g. ``ui_text``, ``ui_dropdown``)
return marimo UI elements whose default value is the stored parameter for a
given key.
"""

import json
from pathlib import Path
from typing import Any, Sequence
import marimo as mo


class Parameters:
    """
    Persist a set of simple key-value parameters to a JSON file.

    The JSON file is read on initialization (if it exists), and any updates
    made via :meth:`set` are immediately written back to disk.
    """

    def __init__(self, filepath: Path | str) -> None:
        """
        Initialize a Parameters store for the given JSON file.

        Parameters
        ----------
        filepath:
            Path to the JSON file used to persist parameters.
        """
        self._path = Path(filepath)
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load parameters from the JSON file if it exists."""
        if self._path.exists():
            with self._path.open("r") as f:
                self._data = json.load(f)
        else:
            # Ensure parent directory exists, but don't create the file yet.
            if self._path.parent:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            self._data = {}

    def _save(self) -> None:
        """Write the current parameters to the JSON file."""
        if self._path.parent:
            self._path.parent.mkdir(parents=True, exist_ok=True)

        with self._path.open("w") as f:
            json.dump(self._data, f, indent=2)

    def get(self, key: str, default: Any | None = None) -> Any:
        """
        Retrieve a parameter value.

        Parameters
        ----------
        key:
            Parameter name.
        default:
            Value to return if the key is not present.
        """
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a parameter value and persist the change to disk.

        The value must be JSON-serializable.

        Parameters
        ----------
        key:
            Parameter name.
        value:
            JSON-serializable value to store.
        """
        # Eagerly check that the value is JSON-serializable to fail fast
        # rather than on save.
        try:
            json.dumps({key: value})
        except TypeError as exc:
            raise TypeError(
                f"Value for key '{key}' is not JSON-serializable: {value!r}"
            ) from exc

        self._data[key] = value
        self._save()

    # --- Marimo UI helpers (default value = parameter for the given key) ---

    def ui_text(self, key: str, **kwargs: Any) -> Any:
        """
        Return a marimo ``ui.text`` element whose default value is this
        parameter store's value for ``key``. Changes are persisted via
        ``on_change`` calling :meth:`set` with ``key``.
        """

        user_on_change = kwargs.pop("on_change", None)

        def _on_change(v: str) -> None:
            self.set(key, v)
            if user_on_change is not None:
                user_on_change(v)

        if key in self._data:
            value = self.get(key)
            value = value if isinstance(value, str) else str(value)
            kwargs.pop("value")
        else:
            value = kwargs.pop("value", "")
        return mo.ui.text(value=value, on_change=_on_change, **kwargs)

    def ui_dropdown(
        self,
        key: str,
        options: Sequence[Any] | dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """
        Return a marimo ``ui.dropdown`` element whose default selected value
        is this parameter store's value for ``key``. ``options`` is passed
        through to ``marimo.ui.dropdown``. Changes are persisted via
        ``on_change`` calling :meth:`set` with ``key``.
        """

        user_on_change = kwargs.pop("on_change", None)

        def _on_change(v: Any) -> None:
            self.set(key, v)
            if user_on_change is not None:
                user_on_change(v)

        if key in self._data:
            value = self.get(key)
        else:
            value = kwargs.pop("value", None)

        if value not in options:
            value = None

        return mo.ui.dropdown(
            options=options, value=value, on_change=_on_change, **kwargs
        )

    def ui_multiselect(
        self,
        key: str,
        options: Sequence[Any] | dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """
        Return a marimo ``ui.multiselect`` element whose default selected
        values are this parameter store's value for ``key`` (a list).
        ``options`` is passed through to ``marimo.ui.multiselect``.
        Changes are persisted via ``on_change`` calling :meth:`set` with ``key``.
        """

        user_on_change = kwargs.pop("on_change", None)

        def _on_change(v: list[object]) -> None:
            self.set(key, v)
            if user_on_change is not None:
                user_on_change(v)

        if key in self._data:
            value = self.get(key)
        else:
            value = kwargs.pop("value", None)
 
        if value is not None and not isinstance(value, (list, tuple)):
            value = [value]

        if value is not None:
            value = [v for v in value if v in options]
        else:
            value = []
 
        return mo.ui.multiselect(
            options=options, value=value, on_change=_on_change, **kwargs
        )

    def ui_text_area(self, key: str, **kwargs: Any) -> Any:
        """
        Return a marimo ``ui.text_area`` element whose default value is this
        parameter store's value for ``key``. Changes are persisted via
        ``on_change`` calling :meth:`set` with ``key``.
        """

        user_on_change = kwargs.pop("on_change", None)

        def _on_change(v: str) -> None:
            self.set(key, v)
            if user_on_change is not None:
                user_on_change(v)

        if key in self._data:
            value = self.get(key)
        else:
            value = kwargs.pop("value", "")

        value = value if isinstance(value, str) else str(value)

        return mo.ui.text_area(
            value=value, on_change=_on_change, **kwargs
        )

    def ui_checkbox(self, key: str, **kwargs: Any) -> Any:
        """
        Return a marimo ``ui.checkbox`` element whose default checked state
        is this parameter store's value for ``key`` (coerced to bool).
        Changes are persisted via ``on_change`` calling :meth:`set` with ``key``.
        """

        user_on_change = kwargs.pop("on_change", None)

        def _on_change(v: bool) -> None:
            self.set(key, v)
            if user_on_change is not None:
                user_on_change(v)

        if key in self._data:
            value = bool(self.get(key, False))
        else:
            value = bool(kwargs.pop("value", False))

        if value is not None:
            value = bool(value)
        else:
            value = False

        return mo.ui.checkbox(
            value=value, on_change=_on_change, **kwargs
        )

