"""DOM types for browser-use integration.

This module provides simplified DOM types focused on what's needed
for ChatBrowserUse integration - primarily element bounding boxes
and a serialized text representation.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


class NodeType(IntEnum):
    """DOM node types based on the DOM specification."""

    ELEMENT_NODE = 1
    ATTRIBUTE_NODE = 2
    TEXT_NODE = 3
    CDATA_SECTION_NODE = 4
    ENTITY_REFERENCE_NODE = 5
    ENTITY_NODE = 6
    PROCESSING_INSTRUCTION_NODE = 7
    COMMENT_NODE = 8
    DOCUMENT_NODE = 9
    DOCUMENT_TYPE_NODE = 10
    DOCUMENT_FRAGMENT_NODE = 11
    NOTATION_NODE = 12


@dataclass
class DOMRect:
    """Bounding box coordinates for a DOM element."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center_x(self) -> float:
        """Get the center X coordinate."""
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        """Get the center Y coordinate."""
        return self.y + self.height / 2

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class DOMElement:
    """A simplified DOM element with essential properties for browser automation."""

    # Core identification
    index: int  # 1-based index for browser-use model
    backend_node_id: int  # Chrome's backend node ID

    # Element info
    tag_name: str
    node_type: NodeType = NodeType.ELEMENT_NODE
    text_content: str = ""
    attributes: dict[str, str] = field(default_factory=dict)

    # Accessibility info
    role: Optional[str] = None
    name: Optional[str] = None  # Accessible name

    # Position
    bounds: Optional[DOMRect] = None
    is_visible: bool = True
    is_interactive: bool = False

    # Hierarchy
    parent_index: Optional[int] = None
    children_indices: list[int] = field(default_factory=list)

    @property
    def center(self) -> Optional[tuple[int, int]]:
        """Get the center coordinates for clicking."""
        if self.bounds:
            return (int(self.bounds.center_x), int(self.bounds.center_y))
        return None

    def get_attribute(self, name: str, default: str = "") -> str:
        """Get an attribute value."""
        return self.attributes.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "backend_node_id": self.backend_node_id,
            "tag_name": self.tag_name,
            "text_content": self.text_content,
            "attributes": self.attributes,
            "role": self.role,
            "name": self.name,
            "bounds": self.bounds.to_dict() if self.bounds else None,
            "is_visible": self.is_visible,
            "is_interactive": self.is_interactive,
        }


@dataclass
class DOMState:
    """The complete DOM state for a page."""

    # All elements indexed for browser-use
    elements: dict[int, DOMElement] = field(default_factory=dict)  # index -> element

    # Quick lookup by backend_node_id
    _by_backend_id: dict[int, DOMElement] = field(default_factory=dict)

    # Page info
    url: str = ""
    title: str = ""

    def add_element(self, element: DOMElement) -> None:
        """Add an element to the state."""
        self.elements[element.index] = element
        self._by_backend_id[element.backend_node_id] = element

    def get_by_index(self, index: int) -> Optional[DOMElement]:
        """Get element by its browser-use index."""
        return self.elements.get(index)

    def get_by_backend_id(self, backend_node_id: int) -> Optional[DOMElement]:
        """Get element by Chrome's backend node ID."""
        return self._by_backend_id.get(backend_node_id)

    def get_coordinates_for_index(self, index: int) -> Optional[tuple[int, int]]:
        """Get click coordinates for an element index."""
        element = self.get_by_index(index)
        if element:
            return element.center
        return None

    def serialize(self, include_attributes: Optional[list[str]] = None) -> str:
        """Serialize the DOM state to browser-use format.

        Output format:
        [1]<button>Submit</button>
        [2]<input type=text placeholder=Email />
        [3]<a href=/about>About Us</a>
        """
        if include_attributes is None:
            include_attributes = [
                "type",
                "name",
                "placeholder",
                "value",
                "href",
                "aria-label",
                "role",
                "checked",
                "selected",
                "disabled",
            ]

        lines = []
        for index in sorted(self.elements.keys()):
            element = self.elements[index]
            if not element.is_interactive:
                continue

            # Build attributes string
            attrs_str = ""
            for attr in include_attributes:
                if attr in element.attributes:
                    value = element.attributes[attr]
                    if value:
                        # Truncate long values
                        if len(value) > 50:
                            value = value[:47] + "..."
                        attrs_str += f" {attr}={value}"

            # Build element representation
            tag = element.tag_name.lower()
            text = element.text_content.strip() if element.text_content else ""

            # Truncate long text
            if len(text) > 100:
                text = text[:97] + "..."

            if text:
                line = f"[{index}]<{tag}{attrs_str}>{text}</{tag}>"
            else:
                line = f"[{index}]<{tag}{attrs_str} />"

            lines.append(line)

        return "\n".join(lines)


# Default attributes to include in serialization (matches browser-use)
DEFAULT_INCLUDE_ATTRIBUTES = [
    "title",
    "type",
    "checked",
    "id",
    "name",
    "role",
    "value",
    "placeholder",
    "alt",
    "aria-label",
    "aria-expanded",
    "aria-checked",
    "href",
    "disabled",
    "required",
]
