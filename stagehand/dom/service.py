"""DOM extraction service using CDP.

This service extracts the DOM state from a page using Chrome DevTools Protocol,
building a representation compatible with browser-use's expected format.
"""

import logging
from typing import Any, Optional

from playwright.async_api import Page

from .views import DEFAULT_INCLUDE_ATTRIBUTES, DOMElement, DOMRect, DOMState, NodeType

logger = logging.getLogger(__name__)

# Elements that are typically interactive
INTERACTIVE_TAGS = {
    "a",
    "button",
    "input",
    "select",
    "textarea",
    "details",
    "summary",
    "label",
}

# Interactive roles
INTERACTIVE_ROLES = {
    "button",
    "link",
    "checkbox",
    "radio",
    "textbox",
    "combobox",
    "listbox",
    "menu",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "tab",
    "switch",
    "slider",
    "spinbutton",
    "searchbox",
    "treeitem",
}

# Elements to skip
SKIP_TAGS = {"script", "style", "head", "meta", "link", "noscript", "template"}


class DomService:
    """Service for extracting DOM state via CDP."""

    def __init__(self, page: Page, logger: Optional[logging.Logger] = None):
        self.page = page
        self.logger = logger or logging.getLogger(__name__)
        self._cdp_session = None

    async def _get_cdp_session(self):
        """Get or create a CDP session."""
        if self._cdp_session is None:
            self._cdp_session = await self.page.context.new_cdp_session(self.page)
        return self._cdp_session

    async def get_dom_state(self) -> DOMState:
        """Extract the current DOM state from the page.

        Returns:
            DOMState with indexed interactive elements and their coordinates.
        """
        try:
            cdp = await self._get_cdp_session()

            # Get accessibility tree for role/name info
            ax_tree = await cdp.send("Accessibility.getFullAXTree")

            # Get DOM snapshot with layout info
            snapshot = await cdp.send(
                "DOMSnapshot.captureSnapshot",
                {
                    "computedStyles": ["display", "visibility", "opacity"],
                    "includeDOMRects": True,
                    "includePaintOrder": True,
                },
            )

            # Build the DOM state
            state = DOMState(
                url=self.page.url,
                title=await self.page.title(),
            )

            # Process the snapshot to build elements
            await self._process_snapshot(state, snapshot, ax_tree)

            return state

        except Exception as e:
            self.logger.error(f"Failed to get DOM state: {e}")
            # Return empty state on error
            return DOMState(url=self.page.url)

    async def _process_snapshot(
        self,
        state: DOMState,
        snapshot: dict[str, Any],
        ax_tree: dict[str, Any],
    ) -> None:
        """Process DOM snapshot and build element list."""
        documents = snapshot.get("documents", [])
        if not documents:
            return

        doc = documents[0]
        nodes = doc.get("nodes", {})
        layout = doc.get("layout", {})
        text_values = doc.get("textValue", {})

        # Get string table
        strings = snapshot.get("strings", [])

        # Build AX node lookup by backend_node_id
        ax_lookup = self._build_ax_lookup(ax_tree)

        # Node arrays from snapshot
        node_names = nodes.get("nodeName", [])
        node_types = nodes.get("nodeType", [])
        node_values = nodes.get("nodeValue", [])
        backend_node_ids = nodes.get("backendNodeId", [])
        attributes = nodes.get("attributes", [])
        parent_indices = nodes.get("parentIndex", [])

        # Layout arrays
        layout_node_indices = layout.get("nodeIndex", [])
        layout_bounds = layout.get("bounds", [])

        # Build bounds lookup by node index
        bounds_by_node = {}
        for i, node_idx in enumerate(layout_node_indices):
            if i < len(layout_bounds):
                bounds = layout_bounds[i]
                if len(bounds) >= 4:
                    bounds_by_node[node_idx] = DOMRect(
                        x=bounds[0],
                        y=bounds[1],
                        width=bounds[2],
                        height=bounds[3],
                    )

        # Text value lookup
        text_value_indices = text_values.get("index", [])
        text_value_values = text_values.get("value", [])
        text_by_node = {}
        for i, node_idx in enumerate(text_value_indices):
            if i < len(text_value_values):
                str_idx = text_value_values[i]
                if str_idx >= 0 and str_idx < len(strings):
                    text_by_node[node_idx] = strings[str_idx]

        # Index counter for interactive elements
        element_index = 1

        # Process each node
        for node_idx in range(len(node_names)):
            name_idx = node_names[node_idx]
            if name_idx < 0 or name_idx >= len(strings):
                continue

            tag_name = strings[name_idx].lower()

            # Skip non-element nodes and unwanted tags
            node_type = node_types[node_idx] if node_idx < len(node_types) else 0
            if node_type != NodeType.ELEMENT_NODE:
                continue
            if tag_name in SKIP_TAGS:
                continue

            # Get backend node ID
            backend_id = (
                backend_node_ids[node_idx]
                if node_idx < len(backend_node_ids)
                else node_idx
            )

            # Get bounds
            bounds = bounds_by_node.get(node_idx)

            # Skip elements without bounds (not visible)
            if not bounds or bounds.width <= 0 or bounds.height <= 0:
                continue

            # Parse attributes
            element_attrs = {}
            if node_idx < len(attributes):
                attr_indices = attributes[node_idx]
                for i in range(0, len(attr_indices), 2):
                    if i + 1 < len(attr_indices):
                        key_idx = attr_indices[i]
                        val_idx = attr_indices[i + 1]
                        if (
                            key_idx >= 0
                            and key_idx < len(strings)
                            and val_idx >= 0
                            and val_idx < len(strings)
                        ):
                            element_attrs[strings[key_idx]] = strings[val_idx]

            # Get accessibility info
            ax_info = ax_lookup.get(backend_id, {})
            role = ax_info.get("role")
            name = ax_info.get("name")

            # Determine if interactive
            is_interactive = self._is_interactive(tag_name, role, element_attrs)

            # Get text content
            text_content = text_by_node.get(node_idx, "")
            if not text_content and name:
                text_content = name

            # Only add interactive elements to the indexed list
            if is_interactive:
                element = DOMElement(
                    index=element_index,
                    backend_node_id=backend_id,
                    tag_name=tag_name,
                    node_type=NodeType.ELEMENT_NODE,
                    text_content=text_content,
                    attributes=element_attrs,
                    role=role,
                    name=name,
                    bounds=bounds,
                    is_visible=True,
                    is_interactive=True,
                    parent_index=(
                        parent_indices[node_idx]
                        if node_idx < len(parent_indices)
                        else None
                    ),
                )
                state.add_element(element)
                element_index += 1

    def _build_ax_lookup(
        self, ax_tree: dict[str, Any]
    ) -> dict[int, dict[str, Any]]:
        """Build a lookup from backend_node_id to AX node info."""
        lookup = {}
        nodes = ax_tree.get("nodes", [])

        for node in nodes:
            backend_id = node.get("backendDOMNodeId")
            if backend_id is None:
                continue

            role_obj = node.get("role", {})
            role = role_obj.get("value") if isinstance(role_obj, dict) else None

            name_obj = node.get("name", {})
            name = name_obj.get("value") if isinstance(name_obj, dict) else None

            lookup[backend_id] = {
                "role": role,
                "name": name,
                "ignored": node.get("ignored", False),
            }

        return lookup

    def _is_interactive(
        self,
        tag_name: str,
        role: Optional[str],
        attributes: dict[str, str],
    ) -> bool:
        """Determine if an element is interactive."""
        # Check tag
        if tag_name in INTERACTIVE_TAGS:
            return True

        # Check role
        if role and role.lower() in INTERACTIVE_ROLES:
            return True

        # Check for click handlers
        if any(
            attr.startswith("on") for attr in attributes
        ):
            return True

        # Check for tabindex
        tabindex = attributes.get("tabindex", "")
        if tabindex and tabindex != "-1":
            return True

        # Check for contenteditable
        if attributes.get("contenteditable") == "true":
            return True

        return False

    async def close(self) -> None:
        """Close the CDP session."""
        if self._cdp_session:
            try:
                await self._cdp_session.detach()
            except Exception:
                pass
            self._cdp_session = None
