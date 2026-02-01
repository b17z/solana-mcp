"""Pydantic models for MCP tool input validation."""

from pydantic import BaseModel, Field, field_validator


class SearchInput(BaseModel):
    """Input validation for search operations."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum results")
    source_type: str | None = Field(None, description="Filter by source type")

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str | None) -> str | None:
        if v is None:
            return v
        valid_types = ["rust", "simd", "c", "docs"]
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid source_type '{v}'. Must be one of: {valid_types}")
        return v.lower()


class ConstantLookupInput(BaseModel):
    """Input validation for constant lookup."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Z][A-Z0-9_]*$",
        description="Constant name (UPPER_SNAKE_CASE)",
    )


class FunctionLookupInput(BaseModel):
    """Input validation for function lookup."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z_][a-z0-9_]*$",
        description="Function name (snake_case)",
    )


class GuidanceInput(BaseModel):
    """Input validation for expert guidance lookup."""

    topic: str = Field(..., min_length=1, max_length=50, description="Guidance topic")


class ClientLookupInput(BaseModel):
    """Input validation for client lookup."""

    name: str = Field(..., min_length=1, max_length=50, description="Client name")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.lower().strip()
