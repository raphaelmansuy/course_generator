<!-- Generated: 2025-06-19 00:28:38 UTC -->

# Deployment

## Overview

The AI Course Generator supports multiple deployment approaches through Poetry packaging, direct pip installation, and containerized deployment. The system generates distributable wheel packages and supports both development and production installations with comprehensive dependency management and optional system-level requirements for document generation capabilities.

## Package Types

**Poetry Package Distribution:**
- **Wheel Package** - `poetry build` creates `.whl` in `dist/` directory
- **Source Distribution** - `.tar.gz` archive for source-based installation
- **PyPI Publishing** - `poetry publish` for package repository distribution

**Installation Methods:**
```bash
# Poetry-based installation
pip install dist/ai_course_generator-0.3.0-py3-none-any.whl

# Development installation
pip install -e .

# Direct from repository
pip install git+https://github.com/raphaelmansuy/course_generator.git
```

**Script Entry Points** (pyproject.toml:34-36):
- `generate_course` - Primary CLI command
- `ai-course-generator` - Alternative CLI alias

## Platform Deployment

**System Dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install pandoc texlive-latex-extra

# macOS
brew install pandoc
brew install --cask mactex

# Windows
# Download Pandoc from GitHub releases
# Install MiKTeX or TeX Live distribution
```

**Docker Deployment:**
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pandoc \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

# Install Python package
COPY dist/ai_course_generator-*.whl /tmp/
RUN pip install /tmp/ai_course_generator-*.whl

# Set entry point
ENTRYPOINT ["generate_course"]
```

**Production Configuration:**
- **Environment Variables** - Configure API keys for LiteLLM models
- **Output Directories** - Ensure write permissions for course generation
- **Resource Limits** - Consider memory requirements for large course generation
- **Dependency Validation** - Verify Pandoc and LaTeX installation

## Reference

**Build Commands:**
```bash
# Create distribution packages
poetry build

# Install from source
poetry install --no-dev

# Create requirements file
poetry export -f requirements.txt --output requirements.txt
```

**Distribution Files:**
- `dist/ai_course_generator-0.3.0-py3-none-any.whl` - Wheel package
- `dist/ai_course_generator-0.3.0.tar.gz` - Source distribution
- `requirements.txt` - Alternative dependency specification

**Server Deployment:**
```bash
# Production installation
pip install ai_course_generator[production]

# Service configuration
generate_course --subject "Production Course" --target-directory /var/courses/

# Batch processing
for subject in "Python" "JavaScript" "Docker"; do
    generate_course --subject "$subject" --level intermediate --target-directory /var/courses/
done
```

**Output Locations:**
- **Default Directory** - `./courses/[subject-name]/`
- **Generated Files** - `course.md`, `course.pdf`, `course.docx`, `outline.md`
- **Custom Directories** - Configurable via `--target-directory` parameter

**Performance Considerations:**
- **Memory Usage** - Large courses with many chapters may require 2-4GB RAM
- **Storage Requirements** - PDF generation with LaTeX can create large files
- **API Rate Limits** - Consider AI model provider limitations for batch processing
- **Concurrent Processing** - AsyncIO enables parallel chapter generation

**Troubleshooting Deployment:**
- **Missing System Dependencies** - Verify Pandoc and LaTeX installation
- **Permission Errors** - Ensure write access to target directories
- **API Authentication** - Configure LiteLLM environment variables
- **Resource Exhaustion** - Monitor memory usage during large course generation