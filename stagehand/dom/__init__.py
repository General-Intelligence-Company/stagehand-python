"""DOM extraction and serialization for browser-use integration."""

from .service import DomService
from .views import DOMElement, DOMRect, DOMState

__all__ = ["DomService", "DOMElement", "DOMRect", "DOMState"]
