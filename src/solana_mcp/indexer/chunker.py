"""Chunk content for embedding.

Handles:
- Rust code (functions, structs, enums)
- SIMD markdown documents
- Documentation markdown

Supports incremental indexing via deterministic chunk IDs.
"""

import re
from dataclasses import dataclass, replace
from pathlib import Path

from .compiler import ExtractedConstant, ExtractedItem
from .manifest import generate_chunk_id


@dataclass
class Chunk:
    """A chunk of content ready for embedding."""

    content: str
    source_type: str  # "rust", "simd", "docs"
    source_file: str
    source_name: str  # function name, SIMD number, doc title
    line_number: int | None
    metadata: dict
    chunk_id: str = ""  # Unique ID for incremental indexing


def chunk_rust_item(item: ExtractedItem, repo_name: str = "agave") -> Chunk:
    """Convert an extracted Rust item into a chunk."""
    # Build content with context
    parts = []

    # Add doc comment if present
    if item.doc_comment:
        parts.append(f"/// {item.doc_comment}")

    # Add attributes
    for attr in item.attributes:
        parts.append(attr)

    # Add the code
    parts.append(item.body)

    content = "\n".join(parts)

    return Chunk(
        content=content,
        source_type="rust",
        source_file=item.file_path,
        source_name=item.name,
        line_number=item.line_number,
        metadata={
            "kind": item.kind,
            "signature": item.signature,
            "visibility": item.visibility,
            "repo": repo_name,
        },
    )


def chunk_rust_constant(const: ExtractedConstant, repo_name: str = "agave") -> Chunk:
    """Convert an extracted constant into a chunk."""
    parts = []

    if const.doc_comment:
        parts.append(f"/// {const.doc_comment}")

    type_str = f": {const.type_annotation}" if const.type_annotation else ""
    parts.append(f"const {const.name}{type_str} = {const.value};")

    content = "\n".join(parts)

    return Chunk(
        content=content,
        source_type="rust",
        source_file=const.file_path,
        source_name=const.name,
        line_number=const.line_number,
        metadata={
            "kind": "constant",
            "value": const.value,
            "type": const.type_annotation,
            "repo": repo_name,
        },
    )


def chunk_simd(file_path: Path, simd_dir: Path) -> list[Chunk]:
    """
    Chunk a SIMD markdown file into sections.

    SIMDs have a standard structure:
    - Title and metadata
    - Abstract
    - Motivation
    - Specification (often the longest)
    - Security Considerations
    - Backwards Compatibility
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    chunks = []
    relative_path = str(file_path.relative_to(simd_dir))

    # Extract SIMD number from filename (e.g., "0326-alpenglow.md" -> "SIMD-0326")
    simd_match = re.match(r"(\d+)-(.+)\.md", file_path.name)
    if simd_match:
        simd_number = f"SIMD-{simd_match.group(1)}"
        simd_name = simd_match.group(2).replace("-", " ").title()
    else:
        simd_number = file_path.stem
        simd_name = file_path.stem

    # Split by headers
    sections = re.split(r"^(#{1,3}\s+.+)$", content, flags=re.MULTILINE)

    current_header = f"# {simd_number}: {simd_name}"
    current_content = []
    current_line = 1

    for i, section in enumerate(sections):
        if re.match(r"^#{1,3}\s+", section):
            # This is a header
            if current_content:
                # Save previous section
                section_text = "\n".join(current_content).strip()
                if section_text and len(section_text) > 50:  # Skip tiny sections
                    chunks.append(
                        Chunk(
                            content=f"{current_header}\n\n{section_text}",
                            source_type="simd",
                            source_file=relative_path,
                            source_name=f"{simd_number} - {current_header.lstrip('#').strip()}",
                            line_number=current_line,
                            metadata={
                                "simd_number": simd_number,
                                "section": current_header.lstrip("#").strip(),
                            },
                        )
                    )
            current_header = section.strip()
            current_content = []
            current_line = content[: content.find(section)].count("\n") + 1
        else:
            current_content.append(section)

    # Don't forget the last section
    if current_content:
        section_text = "\n".join(current_content).strip()
        if section_text and len(section_text) > 50:
            chunks.append(
                Chunk(
                    content=f"{current_header}\n\n{section_text}",
                    source_type="simd",
                    source_file=relative_path,
                    source_name=f"{simd_number} - {current_header.lstrip('#').strip()}",
                    line_number=current_line,
                    metadata={
                        "simd_number": simd_number,
                        "section": current_header.lstrip("#").strip(),
                    },
                )
            )

    return chunks


def chunk_markdown(file_path: Path, base_dir: Path, source_type: str = "docs") -> list[Chunk]:
    """
    Chunk a generic markdown file into sections.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    chunks = []
    relative_path = str(file_path.relative_to(base_dir))

    # Split by headers
    sections = re.split(r"^(#{1,3}\s+.+)$", content, flags=re.MULTILINE)

    current_header = f"# {file_path.stem}"
    current_content = []
    current_line = 1

    for section in sections:
        if re.match(r"^#{1,3}\s+", section):
            if current_content:
                section_text = "\n".join(current_content).strip()
                if section_text and len(section_text) > 50:
                    chunks.append(
                        Chunk(
                            content=f"{current_header}\n\n{section_text}",
                            source_type=source_type,
                            source_file=relative_path,
                            source_name=current_header.lstrip("#").strip(),
                            line_number=current_line,
                            metadata={
                                "section": current_header.lstrip("#").strip(),
                            },
                        )
                    )
            current_header = section.strip()
            current_content = []
            current_line = content[: content.find(section)].count("\n") + 1
        else:
            current_content.append(section)

    if current_content:
        section_text = "\n".join(current_content).strip()
        if section_text and len(section_text) > 50:
            chunks.append(
                Chunk(
                    content=f"{current_header}\n\n{section_text}",
                    source_type=source_type,
                    source_file=relative_path,
                    source_name=current_header.lstrip("#").strip(),
                    line_number=current_line,
                    metadata={
                        "section": current_header.lstrip("#").strip(),
                    },
                )
            )

    return chunks


def chunk_all_simds(simd_dir: Path) -> list[Chunk]:
    """Chunk all SIMDs in a directory."""
    proposals_dir = simd_dir / "proposals"
    if not proposals_dir.exists():
        return []

    chunks = []
    for md_file in proposals_dir.glob("*.md"):
        chunks.extend(chunk_simd(md_file, simd_dir))

    return chunks


def chunk_rust_items(
    items: list[ExtractedItem],
    constants: list[ExtractedConstant],
    repo_name: str = "agave",
) -> list[Chunk]:
    """Chunk all Rust items and constants."""
    chunks = []

    for item in items:
        chunks.append(chunk_rust_item(item, repo_name))

    for const in constants:
        chunks.append(chunk_rust_constant(const, repo_name))

    return chunks


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (approx 4 chars per token)."""
    return len(text) // 4


def split_large_chunk(chunk: Chunk, max_tokens: int = 1000) -> list[Chunk]:
    """Split a chunk that's too large into smaller pieces."""
    if estimate_tokens(chunk.content) <= max_tokens:
        return [chunk]

    # Split by lines, trying to keep logical groups
    lines = chunk.content.split("\n")
    sub_chunks = []
    current_lines = []
    current_tokens = 0

    for line in lines:
        line_tokens = estimate_tokens(line)
        if current_tokens + line_tokens > max_tokens and current_lines:
            # Save current chunk
            sub_content = "\n".join(current_lines)
            sub_chunks.append(
                Chunk(
                    content=sub_content,
                    source_type=chunk.source_type,
                    source_file=chunk.source_file,
                    source_name=f"{chunk.source_name} (part {len(sub_chunks) + 1})",
                    line_number=chunk.line_number,
                    metadata={**chunk.metadata, "part": len(sub_chunks) + 1},
                )
            )
            current_lines = [line]
            current_tokens = line_tokens
        else:
            current_lines.append(line)
            current_tokens += line_tokens

    # Don't forget the last part
    if current_lines:
        sub_content = "\n".join(current_lines)
        sub_chunks.append(
            Chunk(
                content=sub_content,
                source_type=chunk.source_type,
                source_file=chunk.source_file,
                source_name=f"{chunk.source_name} (part {len(sub_chunks) + 1})" if len(sub_chunks) > 0 else chunk.source_name,
                line_number=chunk.line_number,
                metadata={**chunk.metadata, "part": len(sub_chunks) + 1} if len(sub_chunks) > 0 else chunk.metadata,
            )
        )

    return sub_chunks


def chunk_content(
    items: list[ExtractedItem] | None = None,
    constants: list[ExtractedConstant] | None = None,
    simd_dir: Path | None = None,
    docs_dir: Path | None = None,
    repo_name: str = "agave",
    max_tokens: int = 1000,
) -> list[Chunk]:
    """
    Chunk all content for embedding.

    Args:
        items: Extracted Rust items
        constants: Extracted Rust constants
        simd_dir: Directory containing SIMDs
        docs_dir: Directory containing documentation
        repo_name: Name of the source repo
        max_tokens: Maximum tokens per chunk

    Returns:
        List of chunks ready for embedding
    """
    all_chunks = []

    # Chunk Rust items
    if items or constants:
        rust_chunks = chunk_rust_items(
            items or [], constants or [], repo_name
        )
        for chunk in rust_chunks:
            all_chunks.extend(split_large_chunk(chunk, max_tokens))

    # Chunk SIMDs
    if simd_dir and simd_dir.exists():
        simd_chunks = chunk_all_simds(simd_dir)
        for chunk in simd_chunks:
            all_chunks.extend(split_large_chunk(chunk, max_tokens))

    # Chunk docs
    if docs_dir and docs_dir.exists():
        for md_file in docs_dir.glob("**/*.md"):
            doc_chunks = chunk_markdown(md_file, docs_dir, "docs")
            for chunk in doc_chunks:
                all_chunks.extend(split_large_chunk(chunk, max_tokens))

    # Assign chunk IDs
    all_chunks = _assign_chunk_ids(all_chunks)

    return all_chunks


def _assign_chunk_ids(chunks: list[Chunk], project: str = "sol") -> list[Chunk]:
    """
    Assign unique chunk IDs to a list of chunks.

    Groups chunks by source file and assigns sequential IDs within each file.
    """
    # Group chunks by source file
    file_chunks: dict[str, list[tuple[int, Chunk]]] = {}
    for i, chunk in enumerate(chunks):
        key = f"{chunk.source_type}:{chunk.source_file}"
        if key not in file_chunks:
            file_chunks[key] = []
        file_chunks[key].append((i, chunk))

    # Assign IDs within each file
    result = list(chunks)  # Copy to avoid modifying original
    for key, indexed_chunks in file_chunks.items():
        for file_idx, (original_idx, chunk) in enumerate(indexed_chunks):
            chunk_id = generate_chunk_id(
                project=project,
                source_type=chunk.source_type,
                source_file=chunk.source_file,
                chunk_index=file_idx,
                content=chunk.content,
            )
            result[original_idx] = replace(chunk, chunk_id=chunk_id)

    return result


def chunk_single_file(
    file_path: Path,
    file_type: str,
    base_path: Path,
    project: str = "sol",
    max_tokens: int = 1000,
) -> list[Chunk]:
    """
    Chunk a single file and assign chunk IDs.

    This is used for incremental indexing to process individual files.

    Args:
        file_path: Absolute path to the file
        file_type: Type of file ("rust", "simd", "docs")
        base_path: Base path for relative path calculation
        project: Project identifier for chunk IDs
        max_tokens: Maximum tokens per chunk

    Returns:
        List of chunks with assigned chunk IDs
    """
    chunks = []

    if file_type == "simd":
        raw_chunks = chunk_simd(file_path, base_path)
    elif file_type == "docs":
        raw_chunks = chunk_markdown(file_path, base_path, "docs")
    elif file_type == "rust":
        # For Rust, we expect pre-compiled items, not raw files
        # This should be handled via chunk_rust_items
        return []
    else:
        return []

    # Split large chunks
    for chunk in raw_chunks:
        chunks.extend(split_large_chunk(chunk, max_tokens))

    # Assign chunk IDs
    return _assign_chunk_ids(chunks, project)


if __name__ == "__main__":
    # Test chunking
    import sys

    if len(sys.argv) < 2:
        print("Usage: chunker.py <simd_dir>")
        sys.exit(1)

    simd_path = Path(sys.argv[1])
    chunks = chunk_all_simds(simd_path)

    print(f"Chunked {len(chunks)} sections from SIMDs")
    for chunk in chunks[:5]:
        print(f"  - {chunk.source_name}: {len(chunk.content)} chars")
