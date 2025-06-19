<!-- Generated: 2025-06-19 00:28:38 UTC -->

# Files Catalog

## Overview

The AI Course Generator is organized into core application logic, AI prompt templates, build configuration, and documentation components. The codebase follows a clean separation between workflow orchestration, content generation, and document processing, with comprehensive configuration management and template-driven AI interactions for maintainable and extensible educational content creation.

## Core Source Files

**Main Application Logic:**
- `ai_course_generator/generate_course.py` (89 lines) - Typer-based CLI interface with interactive and direct command modes
- `ai_course_generator/course_generator_agent.py` (539 lines) - QuantaLogic Flow workflow implementation, AI integration, and document generation
- `ai_course_generator/mcq_generator_agent.py` (824 lines) - Multiple choice question generation with batch processing and progress tracking
- `ai_course_generator/mermaid_processor.py` (571 lines) - Diagram and flowchart processing utilities
- `ai_course_generator/__init__.py` (empty) - Package initialization

**Data Models and Workflows:**
- `CourseRequest` model (course_generator_agent.py:72-82) - Input validation and parameter management
- `create_workflow()` function (course_generator_agent.py:460) - Main content generation pipeline
- `MCQRequest` model (mcq_generator_agent.py) - Question generation parameters
- Workflow nodes with `@Nodes.define` and `@Nodes.llm_node` decorators throughout core files

## Platform Implementation

**AI Integration:**
- LiteLLM integration for multi-provider AI model support (course_generator_agent.py:47)
- Template-based prompt engineering with Jinja2 Environment (course_generator_agent.py:49)
- Asynchronous content generation with error handling and retry mechanisms

**Document Processing:**
- Pypandoc integration for PDF/DOCX/EPUB conversion (course_generator_agent.py:46)
- LaTeX support with emoji compatibility (ai_course_generator/emoji-support.tex)
- Multi-format output with customizable generation flags

**Workflow Engine:**
- QuantaLogic Flow integration with node-based processing (course_generator_agent.py:47)
- Progress tracking with Rich console output and progress bars
- Event-driven architecture with workflow observers

## Build System

**Poetry Configuration:**
- `pyproject.toml` - Main build configuration with dependencies, scripts, and tool settings
- Package metadata: name="ai-course-generator", version="0.3.0", Python ^3.12 requirement
- Development dependencies: pytest, black, isort, ruff for testing and code quality

**Alternative Installation:**
- `requirements.txt` - Pip-compatible dependency specification with version pinning
- Git-based installation reference to repository at specific commit hash

**CI/CD Pipeline:**
- `.github/workflows/python-tests.yml` - GitHub Actions workflow for Python 3.10-3.13 matrix testing
- Poetry-based dependency installation and package building verification

## Configuration

**AI Prompt Templates:**
- `prompts/course_system_prompt.j2` - Base system prompt for AI context setting
- `prompts/course_title_generation_prompt.j2` - Course title generation instructions
- `prompts/course_outline_prompt.j2` - Structured course outline creation template
- `prompts/course_chapter_prompt.j2` - Individual chapter content generation
- `prompts/course_title_system_prompt.j2` - Title-specific system context

**Supporting Files:**
- `LICENSE` - MIT license specification
- `.gitignore` - Version control exclusions for Python projects
- `poetry.lock` - Locked dependency versions for reproducible builds

## Reference

**File Organization Patterns:**
- **Package Structure** - Single `ai_course_generator/` package with focused modules
- **Template Directory** - Separate `prompts/` for AI interaction templates
- **Documentation** - `docs/` for structured technical documentation
- **Build Artifacts** - `dist/` directory created by Poetry build process

**Naming Conventions:**
- **Snake Case** - Python module and function naming (generate_course, course_generator_agent)
- **Template Extensions** - `.j2` suffix for Jinja2 template files
- **Workflow Nodes** - Descriptive names matching processing steps (validate_input, generate_title)

**Dependency Relationships:**
- **Core Dependencies** - quantalogic, litellm, pydantic for core functionality
- **Document Processing** - pypandoc, rich, aiofiles for output generation
- **Development Tools** - black, isort, ruff, pytest for code quality and testing
- **System Dependencies** - Pandoc and LaTeX for document format conversion