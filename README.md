<!-- Generated: 2025-06-19 00:28:38 UTC -->

# AI Course Generator

AI-powered educational content creation tool using QuantaLogic Flow workflows and multi-provider AI integration. Generates structured courses in PDF, DOCX, EPUB, and Markdown formats with customizable difficulty levels and comprehensive MCQ support.

## Key Entry Points

**Core Files:**
- `ai_course_generator/generate_course.py` - CLI interface with Typer framework
- `ai_course_generator/course_generator_agent.py` - Main workflow engine and AI integration  
- `pyproject.toml` - Build configuration and dependencies

## Quick Build Commands

```bash
# Install dependencies
poetry install

# Generate course
poetry run generate_course --subject "Python" --level beginner

# Alternative installation
pip install -r requirements.txt
```

## Documentation

**LLM-Optimized Technical Reference:**
- **[docs/project-overview.md](docs/project-overview.md)** - Project purpose, technology stack, and platform requirements
- **[docs/architecture.md](docs/architecture.md)** - System organization, workflow components, and data flow with specific file references
- **[docs/build-system.md](docs/build-system.md)** - Build instructions, dependencies, and platform setup for all environments
- **[docs/development.md](docs/development.md)** - Code patterns, development workflows, and implementation examples from codebase
- **[docs/testing.md](docs/testing.md)** - Testing approach, commands, and CI/CD integration
- **[docs/deployment.md](docs/deployment.md)** - Packaging, distribution, and production deployment strategies
- **[docs/files.md](docs/files.md)** - Comprehensive file catalog with descriptions and dependency relationships