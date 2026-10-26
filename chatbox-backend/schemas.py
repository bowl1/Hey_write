"""API and storage models for templates and workspace endpoints."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class WorkspaceResponse(BaseModel):
    tenant_id: str


class Template(BaseModel):
    source: Literal["private", "default"]
    id: str
    title: str
    category: str = "general"
    description: str = ""
    tags: list[str] = []
    style: str = "Formal"
    language: str = "English"
    use_cases: list[str] = []
    structure: list[str] = []
    content: str = ""
    enabled: bool = True
    version: int = 1
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TemplateListResponse(BaseModel):
    templates: list[Template]


class TemplateCreateRequest(BaseModel):
    title: str
    category: str = "general"
    description: str = ""
    tags: list[str] = []
    style: str = "Formal"
    language: str = "English"
    use_cases: list[str] = []
    content: str


class IndexStatus(BaseModel):
    ok: bool
    error: str | None = None


class TemplateCreateResponse(BaseModel):
    template: Template
    index_status: IndexStatus
