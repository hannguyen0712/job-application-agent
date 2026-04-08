"""
orchestrator.py -- Pipeline coordination with validation and tracing.

Runs: Planner -> Retriever -> Summarizer -> Critic (loop) -> Writer
      -> Resume Tailor -> Cover Letter (optional)
Saves: Markdown prep package + tailored resume + cover letter + JSON trace
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict
from typing import Any, Dict, List

from models import (
    PlannerInput, PlannerOutput,
    RetrieverOutput,
    SummarizerOutput,
    CriticOutput,
    WriterInput, WriterOutput,
    ResumeTailorOutput,
    CoverLetterOutput,
)
from agents import (
    PlannerAgent, RetrieverAgent, SummarizerAgent,
    CriticAgent, WriterAgent, ResumeTailorAgent, CoverLetterAgent,
)

logger = logging.getLogger("job_prep")

MAX_CRITIC_ITERATIONS = 2


# -- Run Trace ----------------------------------------------------------------

class RunTrace:
    """Records every agent input/output for debugging and analysis."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def log(self, agent: str, phase: str, data: Any) -> None:
        """Record one trace entry."""
        serializable = data
        if hasattr(data, "__dataclass_fields__"):
            serializable = asdict(data)
        self.entries.append({
            "agent": agent,
            "phase": phase,
            "timestamp": time.time(),
            "elapsed_sec": round(time.time() - self.start_time, 2),
            "data": serializable,
        })

    def save(self, path: str) -> None:
        """Write the full trace to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, default=str)
        logger.info("Trace saved to %s (%d entries)", path, len(self.entries))


# -- Validation helpers -------------------------------------------------------

def _validate_planner(output: PlannerOutput) -> None:
    if not output.company or output.company == "Unknown":
        logger.warning("Planner: could not identify company name")
    if not output.search_queries:
        logger.warning("Planner: no search queries generated")


def _validate_retriever(output: RetrieverOutput) -> None:
    if not output.snippets:
        logger.warning("Retriever: found 0 snippets (company research may be thin)")


def _validate_summarizer(output: SummarizerOutput) -> None:
    if not output.requirements:
        logger.warning("Summarizer: extracted 0 requirements")
    if not output.company_summary:
        logger.warning("Summarizer: missing company summary")


def _validate_critic(output: CriticOutput) -> None:
    if not output.strengths:
        logger.warning("Critic: found 0 strengths (resume may not match well)")


# -- Main pipeline ------------------------------------------------------------

def run_pipeline(
    job_posting: str,
    resume_text: str,
    output_dir: str = "outputs",
    model: str = "gpt-4.1-mini",
    write_cover_letter: bool = True,
) -> str:
    """
    Run the full Job Prep Agent pipeline.

    Parameters
    ----------
    job_posting : str
        The full text of the job posting.
    resume_text : str
        The full text of the candidate's resume.
    output_dir : str
        Directory where the report and trace are saved.
    model : str
        LLM model for all agents.
    write_cover_letter : bool
        If True, also generate a cover letter (default True).

    Returns
    -------
    str
        File path of the saved Markdown prep package.
    """
    total_steps = 7 if write_cover_letter else 6

    os.makedirs(output_dir, exist_ok=True)
    trace = RunTrace()

    # Instantiate agents
    planner = PlannerAgent(model=model)
    retriever = RetrieverAgent()
    summarizer = SummarizerAgent(model=model)
    critic = CriticAgent(model=model)
    writer = WriterAgent(model=model)
    resume_tailor = ResumeTailorAgent(model=model)
    cover_letter_agent = CoverLetterAgent(model=model) if write_cover_letter else None

    # ── Step 1: Plan ─────────────────────────────────────────────────────
    print(f"\n[1/{total_steps}] Planner: parsing job posting...")
    planner_input = PlannerInput(
        job_posting=job_posting,
        resume_text=resume_text,
    )
    trace.log("Planner", "input", planner_input)

    plan = planner.run(planner_input)
    _validate_planner(plan)
    trace.log("Planner", "output", plan)

    print(f"      {plan.role} at {plan.company} ({plan.location})")
    print(f"      Must-have: {', '.join(plan.must_have_skills[:5])}")
    print(f"      Search queries: {len(plan.search_queries)}")

    # ── Steps 2-4: Retrieve -> Summarize -> Critique (with loop) ─────────
    all_snippets = []
    summarizer_output = None
    critic_result = None

    for iteration in range(MAX_CRITIC_ITERATIONS):
        # Step 2: Retrieve
        queries = (
            plan.search_queries if iteration == 0
            else critic_result.requery_queries
        )
        print(f"\n[2/{total_steps}] Retriever: searching ({len(queries)} queries, "
              f"iteration {iteration + 1})...")

        retriever_output = retriever.run(queries)
        all_snippets.extend(retriever_output.snippets)
        _validate_retriever(retriever_output)
        trace.log("Retriever", f"output_iter{iteration}", retriever_output)

        print(f"      Found {len(retriever_output.snippets)} new snippets "
              f"({len(all_snippets)} total)")

        # Step 3: Summarize
        print(f"\n[3/{total_steps}] Summarizer: extracting requirements...")
        summarizer_output = summarizer.run(job_posting, all_snippets)
        _validate_summarizer(summarizer_output)
        trace.log("Summarizer", f"output_iter{iteration}", summarizer_output)

        hard = sum(1 for r in summarizer_output.requirements
                   if r.category == "hard_skill")
        soft = sum(1 for r in summarizer_output.requirements
                   if r.category == "soft_skill")
        flags = sum(1 for r in summarizer_output.requirements
                    if r.category == "flag")
        print(f"      {len(summarizer_output.requirements)} requirements "
              f"({hard} hard, {soft} soft, {flags} flags)")

        # Step 4: Critique
        print(f"\n[4/{total_steps}] Critic: matching resume to requirements...")
        critic_result = critic.run(resume_text, summarizer_output)
        _validate_critic(critic_result)
        trace.log("Critic", f"output_iter{iteration}", critic_result)

        print(f"      {len(critic_result.strengths)} strengths, "
              f"{len(critic_result.gaps)} gaps, "
              f"{len(critic_result.stories)} stories")

        if not critic_result.requery_needed:
            print("      Critic: analysis complete.")
            break
        else:
            print(f"      Critic: needs more info, re-querying with "
                  f"{len(critic_result.requery_queries)} queries...")

    # ── Step 5: Write ────────────────────────────────────────────────────
    print(f"\n[5/{total_steps}] Writer: composing interview prep package...")
    writer_input = WriterInput(
        company=plan.company,
        role=plan.role,
        planner_output=plan,
        summarizer_output=summarizer_output,
        critic_output=critic_result,
    )
    trace.log("Writer", "input", writer_input)

    result = writer.run(writer_input)
    trace.log("Writer", "output", {"markdown_length": len(result.markdown)})

    # ── Step 6: Resume Tailor ────────────────────────────────────────────
    print(f"\n[6/{total_steps}] Resume Tailor: rewriting resume for this role...")
    tailor_result = resume_tailor.run(
        original_resume=resume_text,
        job_posting=job_posting,
        planner_output=plan,
        critic_output=critic_result,
    )
    trace.log("ResumeTailor", "output", tailor_result)

    print(f"      {len(tailor_result.tweaks)} changes made")
    for t in tailor_result.tweaks[:3]:
        print(f"        - [{t.section}] {t.reason[:70]}")
    if len(tailor_result.tweaks) > 3:
        print(f"        ... and {len(tailor_result.tweaks) - 3} more")

    # ── Step 7: Cover Letter (optional) ──────────────────────────────────
    cover_result = None
    if write_cover_letter and cover_letter_agent:
        print(f"\n[7/{total_steps}] Cover Letter: writing personalized letter...")
        cover_result = cover_letter_agent.run(
            resume_text=resume_text,
            job_posting=job_posting,
            planner_output=plan,
            summarizer_output=summarizer_output,
            critic_output=critic_result,
        )
        trace.log("CoverLetter", "output", {
            "letter_length": len(cover_result.cover_letter),
            "tone_notes": cover_result.tone_notes,
        })

    # ── Save outputs ─────────────────────────────────────────────────────
    slug = re.sub(
        r"[^a-z0-9]+", "_",
        f"{plan.company}_{plan.role}".lower()
    )[:50].strip("_")

    report_path = os.path.join(output_dir, f"prep_{slug}.md")
    resume_path = os.path.join(output_dir, f"resume_{slug}.txt")
    changes_path = os.path.join(output_dir, f"resume_changes_{slug}.md")
    cover_path = os.path.join(output_dir, f"cover_letter_{slug}.md")
    trace_path = os.path.join(output_dir, f"trace_{slug}.json")

    # Interview prep
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result.markdown)

    # Tailored resume
    with open(resume_path, "w", encoding="utf-8") as f:
        f.write(tailor_result.tailored_resume)

    # Resume change log
    with open(changes_path, "w", encoding="utf-8") as f:
        f.write(f"# Resume Changes for {plan.role} at {plan.company}\n\n")
        f.write(f"{tailor_result.summary_of_changes}\n\n")
        f.write("## Detailed Changes\n\n")
        for i, t in enumerate(tailor_result.tweaks, 1):
            f.write(f"### {i}. [{t.section}]\n")
            f.write(f"**Why:** {t.reason}\n\n")
            if t.original and t.original != "NEW":
                f.write(f"**Before:**\n> {t.original}\n\n")
            f.write(f"**After:**\n> {t.revised}\n\n")

    # Cover letter (optional)
    if cover_result:
        with open(cover_path, "w", encoding="utf-8") as f:
            f.write(cover_result.cover_letter)
            f.write(f"\n\n---\n*Tone notes: {cover_result.tone_notes}*\n")

    # Trace
    trace.save(trace_path)

    # Summary
    elapsed = time.time() - trace.start_time
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  Interview prep:   {report_path}")
    print(f"  Tailored resume:  {resume_path}")
    print(f"  Change log:       {changes_path}")
    if cover_result:
        print(f"  Cover letter:     {cover_path}")
    print(f"  Agent trace:      {trace_path}")
    print(f"{'='*60}")

    return report_path
