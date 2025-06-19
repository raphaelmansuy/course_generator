<!-- Generated: 2025-06-19 00:28:38 UTC -->

# Project Overview

## Overview

The AI Course Generator is a Python-based educational content creation tool that leverages advanced language models to generate comprehensive courses automatically. Built on the QuantaLogic Flow workflow engine, it transforms simple course parameters (subject, difficulty, chapters) into complete educational materials with structured content, multiple output formats, and integrated MCQ generation capabilities.

The system uses a node-based processing pipeline to orchestrate AI-driven content generation, ensuring consistent quality and format across all generated materials. It supports multiple AI models through LiteLLM integration and can output courses in PDF, DOCX, EPUB, and Markdown formats with professional styling and LaTeX support.

## Key Files

**Main Entry Points:**
- `ai_course_generator/generate_course.py` - CLI interface with Typer, interactive and direct command modes
- `ai_course_generator/course_generator_agent.py` - Core workflow logic and QuantaLogic Flow integration (539 lines)
- `ai_course_generator/mcq_generator_agent.py` - Multiple choice question generation workflow (824 lines)

**Configuration:**
- `pyproject.toml` - Poetry build configuration, dependencies, and tool settings
- `prompts/course_system_prompt.j2` - Main AI system prompt template
- `prompts/course_title_generation_prompt.j2` - Title generation template
- `prompts/course_outline_prompt.j2` - Course outline generation template
- `prompts/course_chapter_prompt.j2` - Chapter content generation template

## Technology Stack

**Core Framework:**
- **QuantaLogic Flow** - Workflow orchestration engine with node-based processing
- **LiteLLM** - Multi-provider AI model integration (Gemini, OpenAI, etc.)
- **Pydantic** - Data validation and settings management (CourseRequest model in course_generator_agent.py:52-63)

**Content Generation:**
- **Jinja2** - Template rendering for AI prompts (Environment setup in course_generator_agent.py:49)
- **AsyncIO** - Asynchronous processing for concurrent AI requests
- **Pypandoc** - Multi-format document conversion (PDF/DOCX/EPUB generation)

**Development Tools:**
- **Poetry** - Dependency management and packaging
- **Typer + Rich** - CLI interface with progress bars and colored output
- **Black/Isort/Ruff** - Code formatting and linting (configured in pyproject.toml:42-53)

## Platform Support

**Requirements:**
- **Python 3.12+** - Minimum version specified in pyproject.toml:10
- **Pandoc** - Required for document format conversion (PDF, DOCX, EPUB)
- **LuaLaTeX** - Required for PDF generation with LaTeX support

**Platform-Specific Files:**
- `ai_course_generator/emoji-support.tex` - LaTeX emoji support configuration for PDF generation
- `.github/workflows/python-tests.yml` - CI/CD pipeline supporting Python 3.10-3.13
- `requirements.txt` - Alternative dependency specification for pip installations

**Output Locations:**
- Default target directory: `./courses/[subject-name]/`
- Generated files: `course.md`, `course.pdf`, `course.docx`, `course.epub`, `outline.md`
- MCQ outputs: Various formats in specified target directory