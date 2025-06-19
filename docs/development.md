<!-- Generated: 2025-06-19 00:28:38 UTC -->

# Development

## Overview

The project follows modern Python development practices with Poetry dependency management, Black code formatting, and comprehensive type hints. Development is structured around QuantaLogic Flow workflow patterns with asynchronous processing, Pydantic data validation, and Jinja2 template-based AI prompt engineering for consistent and maintainable content generation workflows.

## Code Style

**Formatting Configuration** (pyproject.toml:42-48):
- **Black** - Line length 120, Python 3.12 target
- **isort** - Black-compatible import sorting
- **Type Hints** - Comprehensive typing throughout codebase

**Example from course_generator_agent.py:72-82:**
```python
class CourseRequest(BaseModel):
    subject: str
    number_of_chapters: int
    level: str
    words_by_chapter: int
    target_directory: str
    pdf_generation: bool = True
    docx_generation: bool = True
    epub_generation: bool = False
    model_name: str = "gemini/gemini-2.0-flash"
```

**Async/Await Patterns** (course_generator_agent.py:170-181):
```python
@Nodes.define(output="chapter_content")
async def generate_chapter(
    title: str, outline: str, chapter_num: int, 
    words_by_chapter: int, system_prompt: str, model_name: str
) -> str:
    try:
        # Async AI content generation
        return await ai_content_generation(...)
    except Exception as e:
        logger.error(f"Chapter generation failed: {e}")
        raise
```

## Common Patterns

**QuantaLogic Flow Node Definition:**
```python
# From course_generator_agent.py:94-108
@Nodes.llm_node(
    model=MODEL,
    system_prompt="",
    output="title",
    prompt_file="prompts/course_title_generation_prompt.j2",
    max_tokens=100,
)
async def generate_title(subject: str, level: str, model_name: str) -> str:
    return ""  # LLM fills this
```

**Error Handling Pattern** (course_generator_agent.py:213-241):
```python
async def generate_pdf(full_course: str, output_path: Path) -> Optional[str]:
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _generate_pdf_sync, full_course, output_path)
        return result
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return None
```

**Template Loading Pattern** (course_generator_agent.py:359-377):
```python
async def load_system_prompt(subject: str, level: str) -> str:
    prompts_dir = Path(__file__).parent.parent / "prompts"
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template("course_system_prompt.j2")
    return template.render(subject=subject, level=level)
```

## Workflows

**Development Setup:**
```bash
# Clone and setup
git clone https://github.com/raphaelmansuy/course_generator.git
cd course_generator
poetry install

# Development tools
poetry run black .
poetry run isort .
poetry run ruff check .
```

**Adding New Workflow Nodes:**
1. **Define Node Function** - Use `@Nodes.define` or `@Nodes.llm_node` decorators
2. **Add to Workflow** - Include in `create_workflow()` function
3. **Update Transitions** - Configure node dependencies and conditions
4. **Test Integration** - Validate workflow execution

**Prompt Engineering** (prompts/ directory):
- **course_system_prompt.j2** - Base AI system context
- **course_title_generation_prompt.j2** - Title generation instructions
- **course_outline_prompt.j2** - Course structure generation
- **course_chapter_prompt.j2** - Chapter content creation

**CLI Development** (generate_course.py:26-89):
- **Typer Framework** - Command-line interface with Rich integration
- **Interactive Mode** - User-friendly parameter collection
- **Parameter Validation** - Automatic missing parameter detection

## Reference

**File Organization:**
- **ai_course_generator/** - Main package directory
- **prompts/** - Jinja2 templates for AI interactions
- **docs/** - Documentation files
- **pyproject.toml** - Build and tool configuration

**Key Dependencies:**
- **quantalogic** - Workflow engine and node decorators
- **pydantic** - Data validation and settings management
- **litellm** - Multi-provider AI model integration
- **jinja2** - Template rendering for dynamic prompts

**Development Commands:**
- `poetry run python -m ai_course_generator.generate_course` - Run CLI
- `poetry shell` - Activate virtual environment
- `poetry add <package>` - Add new dependency
- `poetry build` - Create distribution packages

**Common Issues:**
- **Import Errors** - Ensure `poetry install` completed successfully
- **Template Not Found** - Verify prompts/ directory structure
- **Model API Errors** - Check LiteLLM configuration and API keys
- **File Path Issues** - Use Path objects for cross-platform compatibility