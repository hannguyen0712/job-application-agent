# Job Prep Agent

A multi-agent AI system that takes a job posting and your resume, then produces personalized interview preparation materials.

Built for the WiCS (Women in Computer Science) workshop at ASU.

## Architecture

```
Job Posting + Resume
        |
    [Planner]     -- Parse posting into structured fields + search queries
        |
    [Retriever]   -- Search Wikipedia + DuckDuckGo for company info (no LLM)
        |
    [Summarizer]  -- Extract categorized requirements + company insights
        |
    [Critic]      -- Match resume to requirements: strengths, gaps, stories
        |           (loops back to Retriever if info is missing)
    [Writer]      -- Produce the final interview prep package
        |
    prep_<company>_<role>.md
    trace_<company>_<role>.json
```

## Setup

```bash
pip install openai requests python-dotenv
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

## Usage

### Quick demo (built-in example)
```bash
python main.py --demo
```

### With your own files
```bash
# Save a job posting to posting.txt
# Save your resume to resume.txt
python main.py --posting posting.txt --resume resume.txt
```

### Options
```bash
python main.py --posting posting.txt --resume resume.txt -v          # verbose
python main.py --posting posting.txt --resume resume.txt --model gpt-4o  # better model
python main.py --posting posting.txt --resume resume.txt --output-dir my_preps/
```

## Output

The agent produces two files in `outputs/`:

- **`prep_<company>_<role>.md`** -- Your interview prep package with:
  - Company brief (with recent specifics)
  - "Why this company" draft answer (tied to YOUR background)
  - Tailored talking points (mapping your resume to their requirements)
  - Gaps and how to address them
  - Questions they will likely ask (with which stories to use)
  - Smart questions to ask them (research-informed)

- **`trace_<company>_<role>.json`** -- Full trace of every agent's input/output with timestamps (for debugging)

## File Structure

```
job-prep-agent/
    models.py          # Typed message schemas (dataclasses)
    utils.py           # LLM wrapper, JSON parsing, web search
    agents.py          # 5 agent classes with system prompts
    orchestrator.py    # Pipeline coordinator with critic loop + tracing
    main.py            # CLI entry point with demo mode
    README.md          # This file
```

## Cost

Using `gpt-4.1-mini` (default), a single run costs approximately $0.01-0.03.
Using `gpt-4o`, approximately $0.10-0.20 per run but with higher quality output.

## Based On

This follows the same multi-agent pattern as the AutoResearch pipeline from CSE 598 (Agentic AI):
- Typed dataclass messages between agents
- JSON between agents, Markdown to humans
- Orchestrator validates schemas and logs traces
- Critic loop for self-correction

*Women in Computer Science | ASU*
