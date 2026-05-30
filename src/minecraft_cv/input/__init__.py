"""OS-input layer: emitter interface, default NullEmitter, and the lazy factory.

``MacInputEmitter`` is intentionally NOT re-exported here so that importing this package
never pulls in pynput / Quartz. Obtain a real emitter via :func:`create_emitter`.
"""

from __future__ import annotations

from minecraft_cv.input.emitter import InputEmitter, NullEmitter, create_emitter

__all__ = ["InputEmitter", "NullEmitter", "create_emitter"]
