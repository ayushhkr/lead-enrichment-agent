"""
Structured output contract for the LLM extraction step.
Using Pydantic gives us both validation and a JSON schema we can hand
straight to the model via tool calling / structured outputs.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TeamMember(BaseModel):
    name: str = Field(description="Full name of the leadership/team member")
    title: Optional[str] = Field(default=None, description="Their role or title")
    linkedin_url: Optional[str] = Field(
        default=None, description="LinkedIn profile URL if discoverable on the page"
    )


class CompanyIntel(BaseModel):
    domain: str = Field(description="The company domain that was analyzed")
    company_overview: str = Field(
        description="A concise 2-sentence summary of what the company does"
    )
    target_audience: List[str] = Field(
        default_factory=list,
        description=(
            "A concise list of 3-6 genuine ICP/customer groups (roles, teams, or "
            "organization types). Combine close categories; exclude keyword dumps, "
            "use cases, and standalone industries unless explicitly customer groups."
        ),
    )
    contact_points: List[str] = Field(
        default_factory=list,
        description="Generic/public emails found on the site (contact@, sales@, support@, etc.)",
    )
    key_team_members: List[TeamMember] = Field(
        default_factory=list, description="Leadership/team members found on the site"
    )
    data_confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Estimated 0.0-1.0 confidence in the completeness/quality of the extracted data",
    )
    sources_used: List[str] = Field(
        default_factory=list, description="Page URLs the extraction actually drew from"
    )


# JSON schema handed to the LLM for tool-calling / structured output.
COMPANY_INTEL_JSON_SCHEMA = CompanyIntel.model_json_schema()
