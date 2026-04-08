"""
models.py -- Typed messages for inter-agent communication.

Every agent communicates through dataclasses. The from_dict() class methods
handle safe deserialization from LLM JSON responses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# -- Planner ------------------------------------------------------------------

@dataclass
class PlannerInput:
    """Raw job posting text + resume text from the user."""
    job_posting: str
    resume_text: str


@dataclass
class PlannerOutput:
    """Structured extraction from the job posting."""
    company: str
    role: str
    location: str
    must_have_skills: List[str]
    nice_to_have_skills: List[str]
    search_queries: List[str]

    @classmethod
    def from_dict(cls, d: dict) -> "PlannerOutput":
        return cls(
            company=d.get("company", "Unknown"),
            role=d.get("role", "Unknown"),
            location=d.get("location", "Unknown"),
            must_have_skills=d.get("must_have_skills", []),
            nice_to_have_skills=d.get("nice_to_have_skills", []),
            search_queries=d.get("search_queries", []),
        )


# -- Retriever ----------------------------------------------------------------

@dataclass
class Snippet:
    """A raw text snippet from a web source."""
    text: str
    url: str
    title: str


@dataclass
class RetrieverOutput:
    """All retrieved company info."""
    snippets: List[Snippet]


# -- Summarizer ---------------------------------------------------------------

@dataclass
class Requirement:
    """A single extracted requirement with category."""
    description: str
    category: str       # "hard_skill", "soft_skill", "culture", "flag"
    importance: str     # "must_have", "nice_to_have", "signal"


@dataclass
class SummarizerOutput:
    """Categorized requirements and company insights."""
    requirements: List[Requirement]
    company_summary: str
    culture_signals: List[str]

    @classmethod
    def from_dict(cls, d: dict) -> "SummarizerOutput":
        reqs = [
            Requirement(
                description=r.get("description", ""),
                category=r.get("category", "hard_skill"),
                importance=r.get("importance", "must_have"),
            )
            for r in d.get("requirements", [])
        ]
        return cls(
            requirements=reqs,
            company_summary=d.get("company_summary", ""),
            culture_signals=d.get("culture_signals", []),
        )


# -- Critic -------------------------------------------------------------------

@dataclass
class StrengthMatch:
    """A resume item that matches a job requirement."""
    requirement: str
    resume_evidence: str
    talking_point: str


@dataclass
class GapAnalysis:
    """A requirement the resume doesn't directly address."""
    requirement: str
    suggestion: str


@dataclass
class StoryIdea:
    """A behavioral interview story mapped from resume experience."""
    likely_question: str
    experience_to_use: str
    key_points: List[str]


@dataclass
class CriticOutput:
    """Resume-to-job matching analysis."""
    strengths: List[StrengthMatch]
    gaps: List[GapAnalysis]
    stories: List[StoryIdea]
    requery_needed: bool
    requery_queries: List[str]

    @classmethod
    def from_dict(cls, d: dict) -> "CriticOutput":
        strengths = []
        for s in d.get("strengths", []):
            strengths.append(StrengthMatch(
                requirement=s.get("requirement", ""),
                resume_evidence=s.get("resume_evidence", ""),
                talking_point=s.get("talking_point", ""),
            ))

        gaps = []
        for g in d.get("gaps", []):
            gaps.append(GapAnalysis(
                requirement=g.get("requirement", ""),
                suggestion=g.get("suggestion", ""),
            ))

        stories = []
        for st in d.get("stories", []):
            stories.append(StoryIdea(
                likely_question=st.get("likely_question", ""),
                experience_to_use=st.get("experience_to_use", ""),
                key_points=st.get("key_points", []),
            ))

        return cls(
            strengths=strengths,
            gaps=gaps,
            stories=stories,
            requery_needed=d.get("requery_needed", False),
            requery_queries=d.get("requery_queries", []),
        )


# -- Writer -------------------------------------------------------------------

@dataclass
class WriterInput:
    """Everything the Writer needs to produce the prep package."""
    company: str
    role: str
    planner_output: PlannerOutput
    summarizer_output: SummarizerOutput
    critic_output: CriticOutput


@dataclass
class WriterOutput:
    """The final interview prep package as Markdown."""
    markdown: str


# -- Resume Tailor ------------------------------------------------------------

@dataclass
class ResumeTweak:
    """A single suggested change to the resume."""
    section: str            # which resume section to change
    original: str           # original text (or "NEW" if adding)
    revised: str            # revised text
    reason: str             # why this change helps


@dataclass
class ResumeTailorOutput:
    """Tailored resume output."""
    tweaks: List[ResumeTweak]
    tailored_resume: str    # the full rewritten resume text
    summary_of_changes: str # brief overview of what changed and why

    @classmethod
    def from_dict(cls, d: dict) -> "ResumeTailorOutput":
        tweaks = []
        for t in d.get("tweaks", []):
            tweaks.append(ResumeTweak(
                section=t.get("section", ""),
                original=t.get("original", ""),
                revised=t.get("revised", ""),
                reason=t.get("reason", ""),
            ))
        return cls(
            tweaks=tweaks,
            tailored_resume=d.get("tailored_resume", ""),
            summary_of_changes=d.get("summary_of_changes", ""),
        )


# -- Cover Letter -------------------------------------------------------------

@dataclass
class CoverLetterOutput:
    """Generated cover letter."""
    cover_letter: str       # the full cover letter text
    tone_notes: str         # explanation of tone/style choices

    @classmethod
    def from_dict(cls, d: dict) -> "CoverLetterOutput":
        return cls(
            cover_letter=d.get("cover_letter", ""),
            tone_notes=d.get("tone_notes", ""),
        )


# -- Helpers ------------------------------------------------------------------

def to_json(obj) -> str:
    """Serialize a dataclass to a JSON string."""
    return json.dumps(asdict(obj), indent=2, default=str)
