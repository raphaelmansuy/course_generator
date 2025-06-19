<!-- Generated: 2025-06-19 00:28:38 UTC -->

# Build System

## Overview

The project uses Poetry as its primary build system, configured through `pyproject.toml` with comprehensive dependency management, development tools, and packaging settings. The build system supports Python 3.12+ with optional pip-based installation via `requirements.txt`, and includes automated CI/CD through GitHub Actions for cross-version compatibility testing.

## Build Workflows

**Poetry-based Development:**
```bash
# Install dependencies
poetry install --no-root

# Build package
poetry build

# Run in development mode
poetry run python -m ai_course_generator.generate_course --help
```

**Alternative pip Installation:**
```bash
# Install from requirements
pip install -r requirements.txt

# Install development dependencies  
pip install -e .
```

**CLI Usage:**
```bash
# Direct execution
python -m ai_course_generator.generate_course --subject "Python" --level beginner

# Poetry script
poetry run generate_course --interactive
```

## Platform Setup

**Core Dependencies** (pyproject.toml:9-21):
- `quantalogic ^0.61.3` - Workflow engine
- `litellm ^1.61.20` - AI model integration  
- `pypandoc ^1.15` - Document conversion
- `typer ^0.15.2` - CLI framework
- `rich ^13.9.4` - Terminal UI
- `jinja2 ^3.1.5` - Template engine

**System Requirements:**
- **Pandoc** - Document format conversion (install via system package manager)
- **LuaLaTeX** - PDF generation with LaTeX support
- **Git** - For development installation from repository (requirements.txt:1)

**Development Tools** (pyproject.toml:23-28):
- `pytest ^8.0.0` - Testing framework
- `black ^24.1.1` - Code formatting (line-length: 120, pyproject.toml:42-44)
- `isort ^5.13.2` - Import sorting (profile: black, pyproject.toml:46-48)
- `ruff ^0.2.1` - Fast linting (target-version: py312, pyproject.toml:50-52)

## Reference

**Build Targets:**
- `poetry build` - Creates wheel and sdist in `dist/`
- `poetry install` - Installs dependencies in virtual environment
- `poetry run` - Executes commands in Poetry environment

**Script Configurations** (pyproject.toml:34-36):
- `generate_course` - Main CLI entry point
- `ai-course-generator` - Alternative CLI alias

**CI/CD Pipeline** (.github/workflows/python-tests.yml):
- **Matrix Testing** - Python versions 3.10, 3.11, 3.12, 3.13
- **Build Verification** - `poetry build` validates package construction
- **Dependency Caching** - Poetry dependencies cached for faster builds

**Troubleshooting:**
- **Missing Pandoc** - Install system package: `apt-get install pandoc` (Ubuntu) or `brew install pandoc` (macOS)
- **LaTeX Issues** - Install TeX distribution: `apt-get install texlive-latex-extra` or download from tex.org
- **Poetry Not Found** - Install via `pip install poetry` or use official installer
- **Permission Errors** - Use `poetry config virtualenvs.create true` for local virtual environments