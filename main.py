#!/usr/bin/env python3
"""
main.py -- CLI entry point for the Job Prep Agent.

Usage:
    # From text files:
    python main.py --posting posting.txt --resume resume.txt

    # With verbose logging:
    python main.py --posting posting.txt --resume resume.txt -v

    # With a different model:
    python main.py --posting posting.txt --resume resume.txt --model gpt-4o

Environment:
    OPENAI_API_KEY must be set (via export or a .env file).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from orchestrator import run_pipeline


def read_file(path: str) -> str:
    """Read a text file and return its contents."""
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


EXAMPLE_POSTING = """
Software Engineer, Backend
Stripe - San Francisco, CA (Hybrid)

About the role:
We're looking for a backend engineer to join our Payments team. You'll design
and build APIs that process billions of dollars in transactions, working with
a team that values reliability, clean code, and thoughtful system design.

What you'll do:
- Design, build, and maintain APIs and backend services
- Write clean, well-tested code in Ruby and Java
- Collaborate with product and design teams to ship features
- Participate in on-call rotation for critical payment systems
- Mentor junior engineers and contribute to engineering culture

What we look for:
- 3+ years of backend engineering experience
- Strong proficiency in Ruby, Java, or similar languages
- Experience designing and building RESTful APIs
- Solid understanding of relational databases (PostgreSQL preferred)
- Experience with distributed systems and microservices
- Strong communication skills and ability to work cross-functionally

Nice to have:
- Experience with payment systems or fintech
- Familiarity with Kubernetes and containerized deployments
- Experience with event-driven architectures (Kafka, RabbitMQ)
- Contributions to open-source projects

Benefits:
- Competitive salary and equity
- Health, dental, and vision insurance
- Flexible PTO and remote work options
- Annual learning and development budget
- Team offsites and company retreats
""".strip()

EXAMPLE_RESUME = """
JANE SMITH
Software Engineer | jane.smith@email.com | github.com/janesmith

EDUCATION
B.S. Computer Science, Arizona State University, 2024
- GPA: 3.8/4.0
- Relevant coursework: Distributed Systems, Database Management,
  Software Engineering, Algorithms

EXPERIENCE

Software Engineering Intern | TechCorp (Summer 2023)
- Built REST APIs in Python/Flask serving 10,000+ daily requests
- Designed PostgreSQL schemas for user analytics pipeline
- Wrote comprehensive test suites achieving 92% code coverage
- Participated in code reviews and agile sprint planning

Research Assistant | ASU Biocomputing Lab (2022-2024)
- Developed data processing pipeline in Python for tumor simulation
- Optimized parallel computation using UPC++ on HPC clusters
- Presented research findings at undergraduate symposium

PROJECTS

Campus Events Platform (React + Node.js + PostgreSQL)
- Full-stack web app with RESTful API backend
- Implemented authentication, search, and notification features
- Deployed on AWS EC2 with Docker containers

Open-Source Contributions
- Contributed bug fixes to 2 Python libraries on GitHub
- Active in ASU's Google Developer Student Club

LEADERSHIP
- President, Women in Computer Science @ ASU
- Fulton Peer Mentor (freshman retention program)
- Organized workshops and career panels for 200+ students

SKILLS
Languages: Python, Java, JavaScript, SQL, C++
Frameworks: Flask, React, Node.js, Express
Databases: PostgreSQL, MongoDB
Tools: Git, Docker, AWS (EC2, S3), Linux
""".strip()


def main():
    parser = argparse.ArgumentParser(
        description="Job Prep Agent: AI-powered interview preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --posting posting.txt --resume resume.txt\n"
            "  python main.py --demo\n"
            "  python main.py --posting posting.txt --resume resume.txt -v\n"
        ),
    )
    parser.add_argument(
        "--posting",
        help="Path to job posting text file",
    )
    parser.add_argument(
        "--resume",
        help="Path to resume text file",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with built-in example posting and resume",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="LLM model (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory (default: outputs/)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-cover-letter",
        action="store_true",
        help="Skip cover letter generation",
    )
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "Export it or create a .env file with:\n"
            "  OPENAI_API_KEY=sk-your-key-here",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get inputs
    if args.demo:
        print("Running demo with example Stripe posting and resume...\n")
        job_posting = EXAMPLE_POSTING
        resume_text = EXAMPLE_RESUME
    elif args.posting and args.resume:
        job_posting = read_file(args.posting)
        resume_text = read_file(args.resume)
    else:
        parser.print_help()
        print("\nERROR: Provide --posting and --resume, or use --demo")
        sys.exit(1)

    write_cl = not args.no_cover_letter

    print("=" * 60)
    print("JOB PREP AGENT")
    print(f"Model: {args.model}")
    print(f"Posting: {len(job_posting)} chars")
    print(f"Resume: {len(resume_text)} chars")
    print(f"Cover letter: {'yes' if write_cl else 'skipped'}")
    print("=" * 60)

    # Run pipeline
    report_path = run_pipeline(
        job_posting=job_posting,
        resume_text=resume_text,
        output_dir=args.output_dir,
        model=args.model,
        write_cover_letter=write_cl,
    )

    return report_path


if __name__ == "__main__":
    main()
