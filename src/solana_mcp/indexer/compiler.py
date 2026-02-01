"""Compile/extract source code into structured JSON for indexing.

Parses source files to extract:
- Rust: pub fn, pub struct, pub enum, const, impl blocks
- C: functions, structs, typedefs, #define macros (for Firedancer)

Uses tree-sitter for robust parsing when available.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# Try to import tree-sitter parsers
TREE_SITTER_RUST = False
TREE_SITTER_C = False

try:
    import tree_sitter_rust as ts_rust
    from tree_sitter import Language, Parser
    TREE_SITTER_RUST = True
except ImportError:
    pass

try:
    import tree_sitter_c as ts_c
    from tree_sitter import Language, Parser
    TREE_SITTER_C = True
except ImportError:
    pass

TREE_SITTER_AVAILABLE = TREE_SITTER_RUST or TREE_SITTER_C


@dataclass
class ExtractedItem:
    """An extracted code item."""

    kind: str  # function, struct, enum, const, impl, type
    name: str
    signature: str  # For functions: full signature; for types: definition line
    body: str  # Full source code
    doc_comment: str | None
    file_path: str
    line_number: int
    visibility: str  # pub, pub(crate), private
    attributes: list[str]  # #[derive(...)] etc.


@dataclass
class ExtractedConstant:
    """An extracted constant."""

    name: str
    value: str
    type_annotation: str | None
    doc_comment: str | None
    file_path: str
    line_number: int


class RustParser:
    """Parse Rust source code to extract definitions."""

    def __init__(self):
        if TREE_SITTER_AVAILABLE:
            self.parser = Parser(Language(ts_rust.language()))
        else:
            self.parser = None

    def parse_file(self, file_path: Path) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Parse a Rust file and extract items."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return [], []

        if self.parser:
            return self._parse_with_tree_sitter(content, str(file_path))
        else:
            return self._parse_with_regex(content, str(file_path))

    def _parse_with_tree_sitter(
        self, content: str, file_path: str
    ) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Parse using tree-sitter for accurate AST."""
        tree = self.parser.parse(bytes(content, "utf-8"))
        items = []
        constants = []

        lines = content.split("\n")

        def get_text(node) -> str:
            return content[node.start_byte : node.end_byte]

        def get_doc_comment(node) -> str | None:
            """Get doc comment preceding a node."""
            # Look for comment nodes before this node
            doc_lines = []
            line = node.start_point[0] - 1

            while line >= 0:
                line_text = lines[line].strip()
                if line_text.startswith("///"):
                    doc_lines.insert(0, line_text[3:].strip())
                    line -= 1
                elif line_text.startswith("//!"):
                    doc_lines.insert(0, line_text[3:].strip())
                    line -= 1
                elif line_text == "" or line_text.startswith("#["):
                    line -= 1
                else:
                    break

            return "\n".join(doc_lines) if doc_lines else None

        def get_attributes(node) -> list[str]:
            """Get attributes preceding a node."""
            attrs = []
            line = node.start_point[0] - 1

            while line >= 0:
                line_text = lines[line].strip()
                if line_text.startswith("#["):
                    attrs.insert(0, line_text)
                    line -= 1
                elif line_text.startswith("///") or line_text.startswith("//!"):
                    line -= 1
                elif line_text == "":
                    line -= 1
                else:
                    break

            return attrs

        def get_visibility(node) -> str:
            """Determine visibility of a node."""
            for child in node.children:
                if child.type == "visibility_modifier":
                    vis_text = get_text(child)
                    if vis_text == "pub":
                        return "pub"
                    elif "crate" in vis_text:
                        return "pub(crate)"
                    else:
                        return vis_text
            return "private"

        def process_node(node):
            if node.type == "function_item":
                visibility = get_visibility(node)
                if visibility.startswith("pub"):
                    name = None
                    signature_parts = []

                    for child in node.children:
                        if child.type == "identifier":
                            name = get_text(child)
                        elif child.type == "parameters":
                            signature_parts.append(get_text(child))
                        elif child.type == "return_type":
                            signature_parts.append(f"-> {get_text(child)}")

                    if name:
                        # Build signature
                        params = signature_parts[0] if signature_parts else "()"
                        ret = signature_parts[1] if len(signature_parts) > 1 else ""
                        signature = f"fn {name}{params} {ret}".strip()

                        items.append(
                            ExtractedItem(
                                kind="function",
                                name=name,
                                signature=signature,
                                body=get_text(node),
                                doc_comment=get_doc_comment(node),
                                file_path=file_path,
                                line_number=node.start_point[0] + 1,
                                visibility=visibility,
                                attributes=get_attributes(node),
                            )
                        )

            elif node.type == "struct_item":
                visibility = get_visibility(node)
                if visibility.startswith("pub"):
                    name = None
                    for child in node.children:
                        if child.type == "type_identifier":
                            name = get_text(child)
                            break

                    if name:
                        items.append(
                            ExtractedItem(
                                kind="struct",
                                name=name,
                                signature=f"struct {name}",
                                body=get_text(node),
                                doc_comment=get_doc_comment(node),
                                file_path=file_path,
                                line_number=node.start_point[0] + 1,
                                visibility=visibility,
                                attributes=get_attributes(node),
                            )
                        )

            elif node.type == "enum_item":
                visibility = get_visibility(node)
                if visibility.startswith("pub"):
                    name = None
                    for child in node.children:
                        if child.type == "type_identifier":
                            name = get_text(child)
                            break

                    if name:
                        items.append(
                            ExtractedItem(
                                kind="enum",
                                name=name,
                                signature=f"enum {name}",
                                body=get_text(node),
                                doc_comment=get_doc_comment(node),
                                file_path=file_path,
                                line_number=node.start_point[0] + 1,
                                visibility=visibility,
                                attributes=get_attributes(node),
                            )
                        )

            elif node.type == "const_item":
                visibility = get_visibility(node)
                name = None
                type_ann = None
                value = None

                for child in node.children:
                    if child.type == "identifier":
                        name = get_text(child)
                    elif child.type == "type_identifier" or child.type.endswith("_type"):
                        type_ann = get_text(child)

                # Extract value from the full text
                full_text = get_text(node)
                if "=" in full_text:
                    value = full_text.split("=", 1)[1].strip().rstrip(";")

                if name:
                    constants.append(
                        ExtractedConstant(
                            name=name,
                            value=value or "",
                            type_annotation=type_ann,
                            doc_comment=get_doc_comment(node),
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                        )
                    )

            elif node.type == "impl_item":
                # Extract impl blocks for types
                type_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        type_name = get_text(child)
                        break
                    elif child.type == "generic_type":
                        type_name = get_text(child)
                        break

                if type_name:
                    items.append(
                        ExtractedItem(
                            kind="impl",
                            name=f"impl {type_name}",
                            signature=f"impl {type_name}",
                            body=get_text(node),
                            doc_comment=get_doc_comment(node),
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                            visibility="pub",  # impl blocks are effectively pub if type is
                            attributes=get_attributes(node),
                        )
                    )

            # Recurse into children
            for child in node.children:
                process_node(child)

        process_node(tree.root_node)
        return items, constants

    def _parse_with_regex(
        self, content: str, file_path: str
    ) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Fallback regex-based parsing when tree-sitter isn't available."""
        items = []
        constants = []
        lines = content.split("\n")

        # Patterns for extraction
        fn_pattern = re.compile(
            r"^(\s*)(pub(?:\([^)]+\))?\s+)?fn\s+(\w+)\s*(<[^>]+>)?\s*\(([^)]*)\)(\s*->\s*[^{]+)?\s*\{",
            re.MULTILINE,
        )
        struct_pattern = re.compile(
            r"^(\s*)(pub(?:\([^)]+\))?\s+)?struct\s+(\w+)", re.MULTILINE
        )
        enum_pattern = re.compile(
            r"^(\s*)(pub(?:\([^)]+\))?\s+)?enum\s+(\w+)", re.MULTILINE
        )
        const_pattern = re.compile(
            r"^(\s*)(pub(?:\([^)]+\))?\s+)?const\s+(\w+)\s*:\s*([^=]+)\s*=\s*([^;]+);",
            re.MULTILINE,
        )

        # Extract functions
        for match in fn_pattern.finditer(content):
            visibility = match.group(2) or ""
            visibility = visibility.strip()
            if not visibility.startswith("pub"):
                continue

            name = match.group(3)
            params = match.group(5)
            ret = match.group(6) or ""

            # Find the full function body
            start = match.start()
            line_num = content[:start].count("\n") + 1

            # Simple brace matching to find end
            brace_count = 1
            idx = match.end()
            while idx < len(content) and brace_count > 0:
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    brace_count -= 1
                idx += 1

            body = content[match.start() : idx]

            # Get doc comment
            doc_comment = self._get_doc_comment_at_line(lines, line_num - 1)

            items.append(
                ExtractedItem(
                    kind="function",
                    name=name,
                    signature=f"fn {name}({params}){ret}".strip(),
                    body=body,
                    doc_comment=doc_comment,
                    file_path=file_path,
                    line_number=line_num,
                    visibility=visibility if visibility else "pub",
                    attributes=[],
                )
            )

        # Extract structs
        for match in struct_pattern.finditer(content):
            visibility = match.group(2) or ""
            visibility = visibility.strip()
            if not visibility.startswith("pub"):
                continue

            name = match.group(3)
            start = match.start()
            line_num = content[:start].count("\n") + 1

            # Find end of struct (either ; or })
            idx = match.end()
            brace_count = 0
            while idx < len(content):
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    if brace_count == 0:
                        idx += 1
                        break
                    brace_count -= 1
                elif content[idx] == ";" and brace_count == 0:
                    idx += 1
                    break
                idx += 1

            body = content[match.start() : idx]
            doc_comment = self._get_doc_comment_at_line(lines, line_num - 1)

            items.append(
                ExtractedItem(
                    kind="struct",
                    name=name,
                    signature=f"struct {name}",
                    body=body,
                    doc_comment=doc_comment,
                    file_path=file_path,
                    line_number=line_num,
                    visibility=visibility if visibility else "pub",
                    attributes=[],
                )
            )

        # Extract enums
        for match in enum_pattern.finditer(content):
            visibility = match.group(2) or ""
            visibility = visibility.strip()
            if not visibility.startswith("pub"):
                continue

            name = match.group(3)
            start = match.start()
            line_num = content[:start].count("\n") + 1

            # Find end of enum
            idx = match.end()
            brace_count = 0
            while idx < len(content):
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    if brace_count == 1:
                        idx += 1
                        break
                    brace_count -= 1
                idx += 1

            body = content[match.start() : idx]
            doc_comment = self._get_doc_comment_at_line(lines, line_num - 1)

            items.append(
                ExtractedItem(
                    kind="enum",
                    name=name,
                    signature=f"enum {name}",
                    body=body,
                    doc_comment=doc_comment,
                    file_path=file_path,
                    line_number=line_num,
                    visibility=visibility if visibility else "pub",
                    attributes=[],
                )
            )

        # Extract constants
        for match in const_pattern.finditer(content):
            visibility = match.group(2) or ""
            if not visibility.strip().startswith("pub"):
                continue

            name = match.group(3)
            type_ann = match.group(4).strip()
            value = match.group(5).strip()

            start = match.start()
            line_num = content[:start].count("\n") + 1
            doc_comment = self._get_doc_comment_at_line(lines, line_num - 1)

            constants.append(
                ExtractedConstant(
                    name=name,
                    value=value,
                    type_annotation=type_ann,
                    doc_comment=doc_comment,
                    file_path=file_path,
                    line_number=line_num,
                )
            )

        return items, constants

    def _get_doc_comment_at_line(self, lines: list[str], line_idx: int) -> str | None:
        """Get doc comment ending at the given line index."""
        doc_lines = []
        idx = line_idx - 1

        while idx >= 0:
            line = lines[idx].strip()
            if line.startswith("///"):
                doc_lines.insert(0, line[3:].strip())
                idx -= 1
            elif line.startswith("//!"):
                doc_lines.insert(0, line[3:].strip())
                idx -= 1
            elif line == "" or line.startswith("#["):
                idx -= 1
            else:
                break

        return "\n".join(doc_lines) if doc_lines else None


class CParser:
    """Parse C source code to extract definitions (for Firedancer)."""

    def __init__(self):
        if TREE_SITTER_C:
            self.parser = Parser(Language(ts_c.language()))
        else:
            self.parser = None

    def parse_file(self, file_path: Path) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Parse a C file and extract items."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return [], []

        if self.parser:
            return self._parse_with_tree_sitter(content, str(file_path))
        else:
            return self._parse_with_regex(content, str(file_path))

    def _parse_with_tree_sitter(
        self, content: str, file_path: str
    ) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Parse using tree-sitter for accurate AST."""
        tree = self.parser.parse(bytes(content, "utf-8"))
        items = []
        constants = []

        lines = content.split("\n")

        def get_text(node) -> str:
            return content[node.start_byte : node.end_byte]

        def get_doc_comment(node) -> str | None:
            """Get doc comment preceding a node (C-style /* */ or //)."""
            doc_lines = []
            line = node.start_point[0] - 1

            while line >= 0:
                line_text = lines[line].strip()
                if line_text.startswith("//"):
                    doc_lines.insert(0, line_text[2:].strip())
                    line -= 1
                elif line_text.endswith("*/"):
                    # Multi-line comment, find start
                    comment_lines = [line_text.rstrip("*/").strip()]
                    line -= 1
                    while line >= 0 and "/*" not in lines[line]:
                        comment_lines.insert(0, lines[line].strip().lstrip("*").strip())
                        line -= 1
                    if line >= 0:
                        start_line = lines[line].strip().lstrip("/*").strip()
                        if start_line:
                            comment_lines.insert(0, start_line)
                    doc_lines = comment_lines + doc_lines
                    break
                elif line_text == "":
                    line -= 1
                else:
                    break

            return "\n".join(doc_lines) if doc_lines else None

        def process_node(node):
            # Function definitions
            if node.type == "function_definition":
                name = None
                return_type = None
                params = None

                for child in node.children:
                    if child.type == "function_declarator":
                        for sub in child.children:
                            if sub.type == "identifier":
                                name = get_text(sub)
                            elif sub.type == "parameter_list":
                                params = get_text(sub)
                    elif child.type in ("primitive_type", "type_identifier", "sized_type_specifier"):
                        return_type = get_text(child)

                if name:
                    signature = f"{return_type or 'void'} {name}{params or '()'}"
                    items.append(
                        ExtractedItem(
                            kind="function",
                            name=name,
                            signature=signature,
                            body=get_text(node),
                            doc_comment=get_doc_comment(node),
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                            visibility="pub",  # C doesn't have visibility modifiers in same way
                            attributes=[],
                        )
                    )

            # Struct definitions
            elif node.type == "struct_specifier":
                name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        name = get_text(child)
                        break

                if name and any(c.type == "field_declaration_list" for c in node.children):
                    items.append(
                        ExtractedItem(
                            kind="struct",
                            name=name,
                            signature=f"struct {name}",
                            body=get_text(node),
                            doc_comment=get_doc_comment(node),
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                            visibility="pub",
                            attributes=[],
                        )
                    )

            # Enum definitions
            elif node.type == "enum_specifier":
                name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        name = get_text(child)
                        break

                if name:
                    items.append(
                        ExtractedItem(
                            kind="enum",
                            name=name,
                            signature=f"enum {name}",
                            body=get_text(node),
                            doc_comment=get_doc_comment(node),
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                            visibility="pub",
                            attributes=[],
                        )
                    )

            # Typedef
            elif node.type == "type_definition":
                name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        name = get_text(child)

                if name:
                    items.append(
                        ExtractedItem(
                            kind="type",
                            name=name,
                            signature=f"typedef {name}",
                            body=get_text(node),
                            doc_comment=get_doc_comment(node),
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                            visibility="pub",
                            attributes=[],
                        )
                    )

            # Recurse into children
            for child in node.children:
                process_node(child)

        process_node(tree.root_node)

        # Also extract #define constants with regex (tree-sitter doesn't handle preprocessor well)
        constants.extend(self._extract_defines(content, file_path))

        return items, constants

    def _extract_defines(self, content: str, file_path: str) -> list[ExtractedConstant]:
        """Extract #define macros."""
        constants = []
        lines = content.split("\n")

        define_pattern = re.compile(r"^#define\s+(\w+)\s+(.+)$")

        for i, line in enumerate(lines):
            match = define_pattern.match(line.strip())
            if match:
                name = match.group(1)
                value = match.group(2).strip()

                # Skip function-like macros (they have parentheses right after name)
                if "(" in name:
                    continue

                # Get preceding comment
                doc_comment = None
                if i > 0:
                    prev_line = lines[i - 1].strip()
                    if prev_line.startswith("//"):
                        doc_comment = prev_line[2:].strip()
                    elif prev_line.endswith("*/"):
                        # Try to get multi-line comment
                        comment_lines = []
                        j = i - 1
                        while j >= 0 and "/*" not in lines[j]:
                            comment_lines.insert(0, lines[j].strip().lstrip("*").strip())
                            j -= 1
                        if comment_lines:
                            doc_comment = "\n".join(comment_lines)

                constants.append(
                    ExtractedConstant(
                        name=name,
                        value=value,
                        type_annotation=None,  # C macros don't have types
                        doc_comment=doc_comment,
                        file_path=file_path,
                        line_number=i + 1,
                    )
                )

        return constants

    def _parse_with_regex(
        self, content: str, file_path: str
    ) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Fallback regex-based parsing when tree-sitter isn't available."""
        items = []
        constants = []
        lines = content.split("\n")

        # Function pattern (simplified)
        fn_pattern = re.compile(
            r"^(\w+(?:\s*\*)?)\s+(\w+)\s*\(([^)]*)\)\s*\{",
            re.MULTILINE,
        )

        # Struct pattern
        struct_pattern = re.compile(
            r"^(?:typedef\s+)?struct\s+(\w+)\s*\{",
            re.MULTILINE,
        )

        # Extract functions
        for match in fn_pattern.finditer(content):
            return_type = match.group(1).strip()
            name = match.group(2)
            params = match.group(3)

            # Skip if it's a control statement
            if name in ("if", "while", "for", "switch"):
                continue

            start = match.start()
            line_num = content[:start].count("\n") + 1

            # Find the full function body (brace matching)
            brace_count = 1
            idx = match.end()
            while idx < len(content) and brace_count > 0:
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    brace_count -= 1
                idx += 1

            body = content[match.start() : idx]

            # Get doc comment
            doc_comment = self._get_doc_comment_at_line(lines, line_num - 1)

            items.append(
                ExtractedItem(
                    kind="function",
                    name=name,
                    signature=f"{return_type} {name}({params})",
                    body=body,
                    doc_comment=doc_comment,
                    file_path=file_path,
                    line_number=line_num,
                    visibility="pub",
                    attributes=[],
                )
            )

        # Extract structs
        for match in struct_pattern.finditer(content):
            name = match.group(1)
            start = match.start()
            line_num = content[:start].count("\n") + 1

            # Find end of struct
            brace_count = 1
            idx = match.end()
            while idx < len(content) and brace_count > 0:
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    brace_count -= 1
                idx += 1

            # Skip past the semicolon
            while idx < len(content) and content[idx] != ";":
                idx += 1
            idx += 1

            body = content[match.start() : idx]
            doc_comment = self._get_doc_comment_at_line(lines, line_num - 1)

            items.append(
                ExtractedItem(
                    kind="struct",
                    name=name,
                    signature=f"struct {name}",
                    body=body,
                    doc_comment=doc_comment,
                    file_path=file_path,
                    line_number=line_num,
                    visibility="pub",
                    attributes=[],
                )
            )

        # Extract #define constants
        constants.extend(self._extract_defines(content, file_path))

        return items, constants

    def _get_doc_comment_at_line(self, lines: list[str], line_idx: int) -> str | None:
        """Get doc comment ending at the given line index."""
        doc_lines = []
        idx = line_idx - 1

        while idx >= 0:
            line = lines[idx].strip()
            if line.startswith("//"):
                doc_lines.insert(0, line[2:].strip())
                idx -= 1
            elif line == "":
                idx -= 1
            else:
                break

        return "\n".join(doc_lines) if doc_lines else None


def compile_c(
    source_dir: Path,
    output_dir: Path,
    file_patterns: list[str] | None = None,
) -> dict:
    """
    Compile C source files into JSON extracts.

    Args:
        source_dir: Directory containing C source files
        output_dir: Directory to write JSON output
        file_patterns: Glob patterns for files to include (default: ["**/*.c", "**/*.h"])

    Returns:
        Statistics about extraction
    """
    if file_patterns is None:
        file_patterns = ["**/*.c", "**/*.h"]

    parser = CParser()
    all_items = []
    all_constants = []

    # Find all C files
    c_files = []
    for pattern in file_patterns:
        c_files.extend(source_dir.glob(pattern))

    # Parse each file
    for file_path in c_files:
        items, constants = parser.parse_file(file_path)

        # Make paths relative to source_dir
        for item in items:
            item.file_path = str(Path(item.file_path).relative_to(source_dir))
        for const in constants:
            const.file_path = str(Path(const.file_path).relative_to(source_dir))

        all_items.extend(items)
        all_constants.extend(constants)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)

    items_file = output_dir / "items.json"
    with open(items_file, "w") as f:
        json.dump([asdict(item) for item in all_items], f, indent=2)

    constants_file = output_dir / "constants.json"
    with open(constants_file, "w") as f:
        json.dump([asdict(const) for const in all_constants], f, indent=2)

    # Build index by name for fast lookup
    index = {
        "functions": {},
        "structs": {},
        "enums": {},
        "constants": {},
        "types": {},
    }

    for item in all_items:
        category = f"{item.kind}s"
        if category in index:
            index[category][item.name] = {
                "file": item.file_path,
                "line": item.line_number,
            }

    for const in all_constants:
        index["constants"][const.name] = {
            "file": const.file_path,
            "line": const.line_number,
            "value": const.value,
        }

    index_file = output_dir / "index.json"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

    return {
        "files_processed": len(c_files),
        "items_extracted": len(all_items),
        "constants_extracted": len(all_constants),
        "functions": len([i for i in all_items if i.kind == "function"]),
        "structs": len([i for i in all_items if i.kind == "struct"]),
        "enums": len([i for i in all_items if i.kind == "enum"]),
        "types": len([i for i in all_items if i.kind == "type"]),
    }


def compile_rust(
    source_dir: Path,
    output_dir: Path,
    file_patterns: list[str] | None = None,
) -> dict:
    """
    Compile Rust source files into JSON extracts.

    Args:
        source_dir: Directory containing Rust source files
        output_dir: Directory to write JSON output
        file_patterns: Glob patterns for files to include (default: ["**/*.rs"])

    Returns:
        Statistics about extraction
    """
    if file_patterns is None:
        file_patterns = ["**/*.rs"]

    parser = RustParser()
    all_items = []
    all_constants = []

    # Find all Rust files
    rust_files = []
    for pattern in file_patterns:
        rust_files.extend(source_dir.glob(pattern))

    # Parse each file
    for file_path in rust_files:
        items, constants = parser.parse_file(file_path)

        # Make paths relative to source_dir
        for item in items:
            item.file_path = str(Path(item.file_path).relative_to(source_dir))
        for const in constants:
            const.file_path = str(Path(const.file_path).relative_to(source_dir))

        all_items.extend(items)
        all_constants.extend(constants)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)

    items_file = output_dir / "items.json"
    with open(items_file, "w") as f:
        json.dump([asdict(item) for item in all_items], f, indent=2)

    constants_file = output_dir / "constants.json"
    with open(constants_file, "w") as f:
        json.dump([asdict(const) for const in all_constants], f, indent=2)

    # Build index by name for fast lookup
    index = {
        "functions": {},
        "structs": {},
        "enums": {},
        "constants": {},
        "impls": {},
    }

    for item in all_items:
        category = f"{item.kind}s"
        if category in index:
            index[category][item.name] = {
                "file": item.file_path,
                "line": item.line_number,
            }

    for const in all_constants:
        index["constants"][const.name] = {
            "file": const.file_path,
            "line": const.line_number,
            "value": const.value,
        }

    index_file = output_dir / "index.json"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

    return {
        "files_processed": len(rust_files),
        "items_extracted": len(all_items),
        "constants_extracted": len(all_constants),
        "functions": len([i for i in all_items if i.kind == "function"]),
        "structs": len([i for i in all_items if i.kind == "struct"]),
        "enums": len([i for i in all_items if i.kind == "enum"]),
        "impls": len([i for i in all_items if i.kind == "impl"]),
    }


def load_compiled_items(compiled_dir: Path) -> list[ExtractedItem]:
    """Load compiled items from JSON."""
    items_file = compiled_dir / "items.json"
    if not items_file.exists():
        return []

    with open(items_file) as f:
        data = json.load(f)

    return [ExtractedItem(**item) for item in data]


def load_compiled_constants(compiled_dir: Path) -> list[ExtractedConstant]:
    """Load compiled constants from JSON."""
    constants_file = compiled_dir / "constants.json"
    if not constants_file.exists():
        return []

    with open(constants_file) as f:
        data = json.load(f)

    return [ExtractedConstant(**item) for item in data]


def lookup_constant(name: str, compiled_dir: Path) -> ExtractedConstant | None:
    """Fast lookup of a constant by name."""
    index_file = compiled_dir / "index.json"
    if not index_file.exists():
        return None

    with open(index_file) as f:
        index = json.load(f)

    if name not in index.get("constants", {}):
        return None

    # Load full constant data
    constants = load_compiled_constants(compiled_dir)
    for const in constants:
        if const.name == name:
            return const

    return None


def lookup_function(name: str, compiled_dir: Path) -> ExtractedItem | None:
    """Fast lookup of a function by name."""
    index_file = compiled_dir / "index.json"
    if not index_file.exists():
        return None

    with open(index_file) as f:
        index = json.load(f)

    if name not in index.get("functions", {}):
        return None

    items = load_compiled_items(compiled_dir)
    for item in items:
        if item.kind == "function" and item.name == name:
            return item

    return None


if __name__ == "__main__":
    # Test compilation
    import sys

    if len(sys.argv) < 3:
        print("Usage: compiler.py <source_dir> <output_dir>")
        sys.exit(1)

    source = Path(sys.argv[1])
    output = Path(sys.argv[2])

    print(f"Compiling Rust from {source} to {output}...")
    stats = compile_rust(source, output)
    print(f"Results: {stats}")
