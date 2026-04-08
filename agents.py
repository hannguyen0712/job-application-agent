"""
Each agent is a class with a run() method that takes a typed input and
returns a typed output. The intelligence comes from the system prompt.
"""
from __future__ import annotations

import logging
from typing import List

from models import (
    PlannerInput, PlannerOutput,
    RetrieverOutput, Snippet,
    SummarizerOutput, Requirement,
    CriticOutput, StrengthMatch, GapAnalysis, StoryIdea,
    WriterInput, WriterOutput,
    ResumeTailorOutput, ResumeTweak,
    CoverLetterOutput,
)
from utils import call_llm, safe_parse, search_wikipedia, search_duckduckgo

logger = logging.getLogger("job_prep")


# == Agent 1: Planner =========================================================

class PlannerAgent:
    """Parses a job posting into structured fields and generates search queries."""

    SYSTEM_PROMPT = (
        "You are a job posting analyst. Given a raw job posting, extract "
        "structured information about the role.\n\n"
        "Respond ONLY with valid JSON in this exact schema:\n"
        "{\n"
        '  "company": "<company name>",\n'
        '  "role": "<job title>",\n'
        '  "location": "<location, or Remote>",\n'
        '  "must_have_skills": ["skill1", "skill2", "skill3"],\n'
        '  "nice_to_have_skills": ["skill1", "skill2"],\n'
        '  "search_queries": [\n'
        '    "<company> engineering culture",\n'
        '    "<company> recent news 2025",\n'
        '    "<company> tech stack",\n'
        '    "<company> employee reviews"\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Extract 3-8 must-have skills (explicitly required in the posting)\n"
        "- Extract 1-5 nice-to-have skills (preferred or bonus)\n"
        "- Generate 3-5 search queries that would help a candidate research "
        "this company for an interview\n"
        "- If a field is unclear, make your best guess from context"
    )

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    def run(self, inp: PlannerInput) -> PlannerOutput:
        """Parse the job posting into structured fields."""
        logger.info("Planner: parsing job posting...")
        raw = call_llm(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=f"Job posting:\n\n{inp.job_posting}",
            model=self.model,
        )
        data = safe_parse(raw, model=self.model)
        result = PlannerOutput.from_dict(data)
        logger.info(
            "Planner: %s at %s (%s), %d must-have, %d nice-to-have, %d queries",
            result.role, result.company, result.location,
            len(result.must_have_skills),
            len(result.nice_to_have_skills),
            len(result.search_queries),
        )
        return result


# == Agent 2: Retriever ========================================================

class RetrieverAgent:
    """
    Fetches company info from free web APIs. No LLM needed.

    This is the only agent that does NOT use the LLM. It takes keywords
    from the Planner, hits real APIs, deduplicates by URL, and returns
    raw Snippet objects. This is what grounds the pipeline in reality.
    """

    def run(self, search_queries: List[str]) -> RetrieverOutput:
        """Search Wikipedia and DuckDuckGo for company info."""
        logger.info("Retriever: searching %d queries...", len(search_queries))
        all_snippets: List[Snippet] = []
        seen_urls: set = set()

        for query in search_queries:
            keywords = query.split()

            # Wikipedia
            for s in search_wikipedia(keywords, max_results=2):
                if s["url"] and s["url"] not in seen_urls:
                    seen_urls.add(s["url"])
                    all_snippets.append(
                        Snippet(text=s["text"], url=s["url"], title=s["title"])
                    )

            # DuckDuckGo
            for s in search_duckduckgo(keywords):
                if s["url"] and s["url"] not in seen_urls:
                    seen_urls.add(s["url"])
                    all_snippets.append(
                        Snippet(text=s["text"], url=s["url"], title=s["title"])
                    )

        logger.info("Retriever: found %d unique snippets.", len(all_snippets))
        return RetrieverOutput(snippets=all_snippets)


# == Agent 3: Summarizer ======================================================

class SummarizerAgent:
    """Extracts categorized requirements and company insights from raw data."""

    SYSTEM_PROMPT = (
        "You are a job requirements analyst. Given a job posting and "
        "company research snippets, extract and categorize the requirements.\n\n"
        "Respond ONLY with valid JSON in this schema:\n"
        "{\n"
        '  "requirements": [\n'
        "    {\n"
        '      "description": "<specific requirement>",\n'
        '      "category": "hard_skill" | "soft_skill" | "culture" | "flag",\n'
        '      "importance": "must_have" | "nice_to_have" | "signal"\n'
        "    }\n"
        "  ],\n"
        '  "company_summary": "<2-3 sentence company overview with recent specifics>",\n'
        '  "culture_signals": ["signal1", "signal2", "signal3"]\n'
        "}\n\n"
        "Categories explained:\n"
        "- hard_skill: specific technologies, languages, frameworks, tools\n"
        "- soft_skill: communication, leadership, collaboration, mentoring\n"
        "- culture: values, work style, team dynamics, company philosophy\n"
        "- flag: red flags (vague requirements, 'wear many hats', high turnover "
        "signals) or green flags (learning budget, growth paths, work-life)\n\n"
        "Rules:\n"
        "- Extract 8-15 requirements across all categories\n"
        "- The company_summary must include at least one RECENT specific "
        "(a product launch, funding round, acquisition, etc.)\n"
        "- Culture signals should be specific observations, not generic praise\n"
        "- If the posting uses phrases like 'fast-paced' or 'wear many hats', "
        "flag them and explain what they might mean"
    )

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    def run(
        self, job_posting: str, snippets: List[Snippet]
    ) -> SummarizerOutput:
        """Extract categorized requirements from posting + research."""
        logger.info("Summarizer: extracting requirements...")

        snippet_text = "\n\n---\n\n".join(
            f"Source: {s.title}\nURL: {s.url}\nContent: {s.text}"
            for s in snippets
        )

        raw = call_llm(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=(
                f"Job posting:\n{job_posting}\n\n"
                f"Company research:\n{snippet_text}"
            ),
            model=self.model,
        )
        data = safe_parse(raw, model=self.model)
        result = SummarizerOutput.from_dict(data)
        logger.info(
            "Summarizer: %d requirements, %d culture signals",
            len(result.requirements), len(result.culture_signals),
        )
        return result


# == Agent 4: Critic ===========================================================

class CriticAgent:
    """
    Cross-references the candidate's resume against job requirements.
    Identifies strengths, gaps, and behavioral stories to prepare.
    """

    SYSTEM_PROMPT = (
        "You are a career coach and interview strategist. Compare a "
        "candidate's resume against job requirements and company context "
        "to produce a match analysis.\n\n"
        "Respond ONLY with valid JSON in this schema:\n"
        "{\n"
        '  "strengths": [\n'
        "    {\n"
        '      "requirement": "<the job requirement>",\n'
        '      "resume_evidence": "<specific resume item that matches>",\n'
        '      "talking_point": "<how to present this in an interview>"\n'
        "    }\n"
        "  ],\n"
        '  "gaps": [\n'
        "    {\n"
        '      "requirement": "<requirement not met>",\n'
        '      "suggestion": "<how to frame this positively or what to learn>"\n'
        "    }\n"
        "  ],\n"
        '  "stories": [\n'
        "    {\n"
        '      "likely_question": "<behavioral question they will ask>",\n'
        '      "experience_to_use": "<specific resume experience>",\n'
        '      "key_points": ["point1", "point2", "point3"]\n'
        "    }\n"
        "  ],\n"
        '  "requery_needed": false,\n'
        '  "requery_queries": []\n'
        "}\n\n"
        "Rules:\n"
        "- Find 3-5 strengths with SPECIFIC resume evidence (not vague)\n"
        "- Find 2-3 gaps and suggest how to frame them constructively\n"
        "- Suggest 2-3 behavioral stories using the STAR framework\n"
        "- Key points should be concrete, not generic\n"
        "- Set requery_needed=true ONLY if critical company info is missing "
        "(e.g., you have no data about the company at all)\n"
        "- Talking points should sound natural, not rehearsed"
    )

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    def run(
        self,
        resume_text: str,
        summarizer_output: SummarizerOutput,
    ) -> CriticOutput:
        """Match resume against requirements and produce gap analysis."""
        logger.info("Critic: matching resume to requirements...")

        req_text = "\n".join(
            f"- [{r.importance}] [{r.category}] {r.description}"
            for r in summarizer_output.requirements
        )

        raw = call_llm(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=(
                f"CANDIDATE RESUME:\n{resume_text}\n\n"
                f"JOB REQUIREMENTS:\n{req_text}\n\n"
                f"COMPANY SUMMARY:\n{summarizer_output.company_summary}\n\n"
                f"CULTURE SIGNALS:\n"
                f"{chr(10).join('- ' + s for s in summarizer_output.culture_signals)}"
            ),
            model=self.model,
        )
        data = safe_parse(raw, model=self.model)
        result = CriticOutput.from_dict(data)
        logger.info(
            "Critic: %d strengths, %d gaps, %d stories, requery=%s",
            len(result.strengths), len(result.gaps),
            len(result.stories), result.requery_needed,
        )
        return result


# == Agent 5: Writer ===========================================================

class WriterAgent:
    """Produces the final interview prep package as Markdown."""

    SYSTEM_PROMPT = (
        "You are an interview preparation coach. Given a complete analysis "
        "of a job posting matched against a candidate's resume, produce a "
        "comprehensive interview prep package.\n\n"
        "Write in Markdown with EXACTLY these sections:\n\n"
        "# Interview Prep: [Role] at [Company]\n\n"
        "## Company Brief\n"
        "2-3 paragraphs. Include founding story, what they do, recent "
        "milestones, and team culture. Be specific enough that the "
        "candidate sounds informed, not rehearsed.\n\n"
        "## Why This Company (Draft Answer)\n"
        "A natural-sounding 3-4 sentence answer that ties the candidate's "
        "SPECIFIC background to the company's SPECIFIC mission and recent "
        "work. Not generic.\n\n"
        "## Your Talking Points\n"
        "3-5 bullet points, each mapping a specific resume experience to "
        "a specific job requirement. Include what to say.\n\n"
        "## Gaps to Address\n"
        "For each gap, a brief script for how to handle it if asked.\n\n"
        "## Questions They Will Likely Ask\n"
        "3-4 behavioral + 2-3 technical questions predicted from the "
        "posting's language. For each, note which story/experience to use "
        "and 2-3 key points to hit.\n\n"
        "## Smart Questions to Ask Them\n"
        "4-5 research-informed questions that demonstrate real preparation. "
        "NOT generic questions like 'what is the culture like'. Reference "
        "specific company initiatives, tech choices, or recent events.\n\n"
        "Rules:\n"
        "- Write in second person ('you') addressing the candidate\n"
        "- Every claim must reference a specific resume item or company fact\n"
        "- Talking points should sound conversational, not bullet-pointy\n"
        "- The whole document should feel like advice from a mentor who "
        "knows both the candidate and the company well"
    )

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    def run(self, inp: WriterInput) -> WriterOutput:
        """Compose the final interview prep package."""
        logger.info("Writer: composing prep package...")

        # Build context for the Writer
        strengths_text = "\n".join(
            f"- REQUIREMENT: {s.requirement}\n"
            f"  EVIDENCE: {s.resume_evidence}\n"
            f"  TALKING POINT: {s.talking_point}"
            for s in inp.critic_output.strengths
        )

        gaps_text = "\n".join(
            f"- GAP: {g.requirement}\n"
            f"  SUGGESTION: {g.suggestion}"
            for g in inp.critic_output.gaps
        )

        stories_text = "\n".join(
            f"- QUESTION: {s.likely_question}\n"
            f"  EXPERIENCE: {s.experience_to_use}\n"
            f"  KEY POINTS: {', '.join(s.key_points)}"
            for s in inp.critic_output.stories
        )

        skills_text = (
            f"Must-have: {', '.join(inp.planner_output.must_have_skills)}\n"
            f"Nice-to-have: {', '.join(inp.planner_output.nice_to_have_skills)}"
        )

        user_prompt = (
            f"COMPANY: {inp.company}\n"
            f"ROLE: {inp.role}\n"
            f"LOCATION: {inp.planner_output.location}\n\n"
            f"SKILLS FROM POSTING:\n{skills_text}\n\n"
            f"COMPANY SUMMARY:\n{inp.summarizer_output.company_summary}\n\n"
            f"CULTURE SIGNALS:\n"
            f"{chr(10).join('- ' + s for s in inp.summarizer_output.culture_signals)}\n\n"
            f"STRENGTHS (resume matches requirements):\n{strengths_text}\n\n"
            f"GAPS (requirements not met):\n{gaps_text}\n\n"
            f"STORIES TO PREPARE:\n{stories_text}"
        )

        markdown = call_llm(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=self.model,
            max_tokens=4096,
        )
        logger.info("Writer: done (%d chars).", len(markdown))
        return WriterOutput(markdown=markdown)


# == Agent 6: Resume Tailor ====================================================

class ResumeTailorAgent:
    """
    Rewrites the candidate's resume to better match the target job posting.

    Does NOT fabricate experience. It reorders, rewords, and emphasizes
    existing content to align with what the posting is looking for.
    """

    SYSTEM_PROMPT = (
        "You are an expert resume writer. Given a candidate's original resume, "
        "a job posting, and an analysis of how the resume matches the posting, "
        "produce a tailored version of the resume.\n\n"
        "Respond ONLY with valid JSON in this schema:\n"
        "{\n"
        '  "tweaks": [\n'
        "    {\n"
        '      "section": "<which section: summary, experience, skills, etc.>",\n'
        '      "original": "<original text, or NEW if adding>",\n'
        '      "revised": "<revised text>",\n'
        '      "reason": "<why this change helps for this specific role>"\n'
        "    }\n"
        "  ],\n"
        '  "tailored_resume": "<the COMPLETE rewritten resume as plain text>",\n'
        '  "summary_of_changes": "<3-5 sentence overview of what changed>"\n'
        "}\n\n"
        "Rules:\n"
        "- NEVER fabricate experience, skills, or credentials the candidate "
        "does not have\n"
        "- NEVER remove truthful information that makes the candidate look bad\n"
        "- DO reorder bullet points to put the most relevant ones first\n"
        "- DO reword bullet points to use keywords from the job posting "
        "(where truthful)\n"
        "- DO add a tailored summary/objective section at the top if missing\n"
        "- DO reorder the skills section to list the posting's required "
        "skills first\n"
        "- DO quantify achievements where possible (if the original has "
        "numbers, keep them; if it says 'improved performance', note that "
        "a number would help)\n"
        "- DO match the posting's language (if they say 'RESTful APIs' "
        "and the resume says 'REST APIs', use their phrasing)\n"
        "- The tailored_resume should be the COMPLETE resume, not just "
        "the changed parts\n"
        "- Produce 4-8 specific tweaks"
    )

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    def run(
        self,
        original_resume: str,
        job_posting: str,
        planner_output: PlannerOutput,
        critic_output: CriticOutput,
    ) -> ResumeTailorOutput:
        """Rewrite the resume to better match the job posting."""
        logger.info("ResumeTailor: tailoring resume...")

        strengths_text = "\n".join(
            f"- Match: {s.requirement} <-> {s.resume_evidence}"
            for s in critic_output.strengths
        )
        gaps_text = "\n".join(
            f"- Gap: {g.requirement} (suggestion: {g.suggestion})"
            for g in critic_output.gaps
        )

        raw = call_llm(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=(
                f"TARGET COMPANY: {planner_output.company}\n"
                f"TARGET ROLE: {planner_output.role}\n\n"
                f"JOB POSTING:\n{job_posting}\n\n"
                f"MUST-HAVE SKILLS: {', '.join(planner_output.must_have_skills)}\n"
                f"NICE-TO-HAVE: {', '.join(planner_output.nice_to_have_skills)}\n\n"
                f"STRENGTHS IDENTIFIED:\n{strengths_text}\n\n"
                f"GAPS IDENTIFIED:\n{gaps_text}\n\n"
                f"ORIGINAL RESUME:\n{original_resume}"
            ),
            model=self.model,
            max_tokens=4096,
        )
        data = safe_parse(raw, model=self.model)
        result = ResumeTailorOutput.from_dict(data)
        logger.info(
            "ResumeTailor: %d tweaks, resume %d chars",
            len(result.tweaks), len(result.tailored_resume),
        )
        return result


# == Agent 7: Cover Letter Writer ==============================================

class CoverLetterAgent:
    """
    Writes a personalized cover letter for the target position.

    Uses company research, the gap analysis, and the candidate's strongest
    matches to compose a letter that reads as genuine and specific.
    """

    SYSTEM_PROMPT = (
        "You are a professional cover letter writer. Given a candidate's "
        "resume, a job posting, company research, and an analysis of how "
        "the candidate matches the role, write a compelling cover letter.\n\n"
        "Respond ONLY with valid JSON in this schema:\n"
        "{\n"
        '  "cover_letter": "<the full cover letter text>",\n'
        '  "tone_notes": "<brief explanation of style choices>"\n'
        "}\n\n"
        "Cover letter structure:\n"
        "- Opening paragraph: hook with a specific reason you want THIS "
        "role at THIS company (not generic). Reference something recent "
        "or specific about the company.\n"
        "- Middle paragraph(s): 2-3 specific experiences from the resume "
        "that directly address the posting's top requirements. Use the "
        "STAR method briefly. Show impact with numbers where available.\n"
        "- Address a gap (optional): if there's a notable gap, briefly "
        "frame it as a growth area or transferable skill.\n"
        "- Closing paragraph: reiterate enthusiasm, connect your "
        "trajectory to the company's direction, call to action.\n\n"
        "Rules:\n"
        "- Length: 250-400 words (3-4 paragraphs). Hiring managers stop "
        "reading after one page.\n"
        "- NEVER use cliches like 'I am writing to express my interest' "
        "or 'I believe I would be a great fit' or 'I am excited to apply'\n"
        "- NEVER fabricate experiences or credentials\n"
        "- DO reference specific company products, values, or recent events\n"
        "- DO match the posting's tone (formal for banks, casual for startups)\n"
        "- DO use the candidate's name and the hiring manager's name if known\n"
        "- The letter should sound like a real person wrote it, not an AI\n"
        "- Include a proper greeting and sign-off"
    )

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    def run(
        self,
        resume_text: str,
        job_posting: str,
        planner_output: PlannerOutput,
        summarizer_output: SummarizerOutput,
        critic_output: CriticOutput,
    ) -> CoverLetterOutput:
        """Write a personalized cover letter."""
        logger.info("CoverLetter: composing letter...")

        strengths_text = "\n".join(
            f"- {s.requirement}: {s.talking_point}"
            for s in critic_output.strengths
        )
        gaps_text = "\n".join(
            f"- {g.requirement}: {g.suggestion}"
            for g in critic_output.gaps
        )

        raw = call_llm(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=(
                f"COMPANY: {planner_output.company}\n"
                f"ROLE: {planner_output.role}\n"
                f"LOCATION: {planner_output.location}\n\n"
                f"JOB POSTING:\n{job_posting}\n\n"
                f"COMPANY SUMMARY:\n{summarizer_output.company_summary}\n\n"
                f"CULTURE SIGNALS:\n"
                f"{chr(10).join('- ' + s for s in summarizer_output.culture_signals)}\n\n"
                f"CANDIDATE STRENGTHS:\n{strengths_text}\n\n"
                f"CANDIDATE GAPS:\n{gaps_text}\n\n"
                f"CANDIDATE RESUME:\n{resume_text}"
            ),
            model=self.model,
            max_tokens=2048,
        )
        data = safe_parse(raw, model=self.model)
        result = CoverLetterOutput.from_dict(data)
        logger.info("CoverLetter: done (%d chars).", len(result.cover_letter))
        return result
