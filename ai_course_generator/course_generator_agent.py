#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru>=0.7.2",
#     "litellm>=1.0.0",
#     "pydantic>=2.0.0",
#     "anyio>=4.0.0",
#     "pypandoc",
#     "typer",
#     "rich",
#     "aiofiles",
#     "jinja2>=3.1.0",
#     "instructor",
#     "quantalogic-flow",
#     "mermaid_processor"
# ]
# ///

"""
Course Generator Agent

This script generates educational course content based on user-defined parameters such as subject,
number_of_chapters, level, and words per chapter. It leverages AI models to produce a course title,
outline, and chapter content, saving the output as Markdown, and optionally as PDF, DOCX, and EPUB.

Requirements:
- **Pandoc**: Required for PDF, DOCX, and EPUB generation.
- **lualatex**: Required for PDF generation using LaTeX.

Usage:
    ./course_generator_agent.py

Output files are saved in the specified target directory (default: ./courses).
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import anyio
import pypandoc
from jinja2 import Environment, FileSystemLoader
from litellm import acompletion
from loguru import logger
from pydantic import BaseModel
from quantalogic_flow.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType

# Default model and parameters
MODEL = "gemini/gemini-2.0-flash"

DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "frequency_penalty": 0.2,
}

OUTLINE_PARAMS = {
    **DEFAULT_PARAMS,
    "temperature": 0.6,
    "max_tokens": 3500,
}

CHAPTER_PARAMS = {
    **DEFAULT_PARAMS,
    "temperature": 0.7,
}


# Course request model
class CourseRequest(BaseModel):
    subject: str
    number_of_chapters: int
    level: str
    words_by_chapter: int
    target_directory: str
    pdf_generation: bool = True
    docx_generation: bool = True
    epub_generation: bool = False
    model_name: str = MODEL


# Utility functions
def calculate_max_tokens(words: int, is_outline: bool = False) -> int:
    """Calculate max tokens based on word count with a buffer."""
    if is_outline:
        base_tokens = 5000
        per_chapter_tokens = 500
        return base_tokens + (per_chapter_tokens * 5)
    return int(words * 4 * 2.0)  # 4 tokens per word + 100% buffer


def check_dependencies():
    """Check if required external tools are installed."""
    if not shutil.which("pandoc"):
        logger.error("Pandoc is not installed or not in PATH. PDF, DOCX, and EPUB generation will fail.")
    if not shutil.which("lualatex"):
        logger.error("lualatex is not installed or not in PATH. PDF generation will fail.")


async def load_template(prompt_file: str, context: Dict[str, Any]) -> str:
    """Load and render a Jinja2 template."""
    env = Environment(loader=FileSystemLoader("prompts"))
    template = env.get_template(prompt_file)
    return template.render(**context)


# Observer for streaming title, outline, and chapters
async def content_stream_observer(event: WorkflowEvent):
    """Observer to stream title, outline, and chapter content as they are generated."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.text import Text

    console = Console()

    if event.event_type == WorkflowEventType.NODE_COMPLETED and event.result is not None:
        if event.node_name == "generate_title":
            console.print(
                Panel(
                    Text(" Course Title Generated", style="bold green"),
                    title="Progress Update",
                    subtitle=f"Generating course: {event.result}",
                    border_style="blue",
                )
            )
        elif event.node_name == "generate_outline":
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            )
            with progress:
                task = progress.add_task("[cyan]Generating Course Outline...", total=1)
                progress.update(task, advance=1)

            console.print(Panel(Text(" Course Outline Generated", style="bold green"), border_style="blue"))
        elif event.node_name == "generate_chapter":
            chapter_num = event.context["completed_chapters"] + 1
            total_chapters = event.context["number_of_chapters"]

            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            )
            with progress:
                task = progress.add_task(
                    f"[cyan]Generating Chapter {chapter_num}/{total_chapters}...", total=total_chapters
                )
                progress.update(task, completed=chapter_num)

            console.print(Panel(Text(f" Chapter {chapter_num} Generated", style="bold green"), border_style="blue"))


# Node definitions
@Nodes.define(output="validation_result")
async def validate_input(subject: str, number_of_chapters: int, level: str, words_by_chapter: int) -> str:
    if not subject:
        raise ValueError("Subject cannot be empty")
    if number_of_chapters < 1:
        raise ValueError("Number of chapters must be at least 1")
    valid_levels = ["beginner", "intermediate", "advanced"]
    if level not in valid_levels:
        raise ValueError(f"Invalid level. Allowed: {', '.join(valid_levels)}")
    if words_by_chapter < 100:
        raise ValueError("Words by chapter must be at least 100")
    return "validation_passed"


@Nodes.llm_node(
    model=MODEL,
    system_prompt="",
    output="title",
    prompt_file="prompts/course_title_generation_prompt.j2",
    max_tokens=100,
)
async def generate_title(subject: str, level: str, model_name: str) -> str:
    return ""  # Placeholder, LLM fills this via decorator


@Nodes.llm_node(
    model=MODEL,
    system_prompt="",
    output="outline",
    prompt_file="prompts/course_outline_prompt.j2",
    **OUTLINE_PARAMS,
)
async def generate_outline(title: str, number_of_chapters: int, level: str, model_name: str) -> str:
    return ""  # Placeholder, LLM fills this


@Nodes.define(output="outline_saved")
async def save_outline(outline: str, title: str, target_directory: str) -> str:
    try:
        path = Path(target_directory).resolve() / "outline.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(outline)
        logger.success(f"Saved outline for '{title}' to {path}")
        return "outline_saved"
    except Exception as e:
        logger.error(f"Failed to save outline: {e}")
        raise


@Nodes.define(output="loop_status")
async def chapter_loop(number_of_chapters: int, completed_chapters: int) -> str:
    if completed_chapters < number_of_chapters:
        return "next"
    return "complete"


async def generate_chapter_content(
    title: str,
    outline: str,
    number_of_chapters: int,
    words_by_chapter: int,
    level: str,
    completed_chapters: int,
    model_name: str,
    system_prompt: str,
) -> str:
    """Generate chapter content with retries and fallback."""
    context = {
        "title": title,
        "outline": outline,
        "number_of_chapters": number_of_chapters,
        "words_by_chapter": words_by_chapter,
        "level": level,
        "completed_chapters": completed_chapters,
    }
    prompt = await load_template("course_chapter_prompt.j2", context)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(3):
        try:
            response = await acompletion(
                model=model_name,
                messages=messages,
                max_tokens=calculate_max_tokens(words_by_chapter),
                **CHAPTER_PARAMS,
            )
            content = response.choices[0].message.content
            if content is None or not content.strip():
                logger.warning(f"Chapter {completed_chapters + 1} content is empty, retrying (attempt {attempt + 1}/3)")
                await asyncio.sleep(1)
                continue
            return content.strip()
        except Exception as e:
            logger.error(f"Failed to generate chapter {completed_chapters + 1} (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                await asyncio.sleep(1)
                continue
            # Fallback content if all retries fail
            logger.error(f"All retries failed for chapter {completed_chapters + 1}, using fallback content")
            return (
                f"# Chapter {completed_chapters + 1}\n\nContent generation failed due to: {e}. Please try again later."
            )
    return f"# Chapter {completed_chapters + 1}\n\nContent generation failed after 3 attempts."


@Nodes.define(output="chapter_content")
async def generate_chapter(
    title: str,
    outline: str,
    number_of_chapters: int,
    words_by_chapter: int,
    level: str,
    completed_chapters: int,
    model_name: str,
    system_prompt: str,
) -> str:
    return await generate_chapter_content(
        title=title,
        outline=outline,
        number_of_chapters=number_of_chapters,
        words_by_chapter=words_by_chapter,
        level=level,
        completed_chapters=completed_chapters,
        model_name=model_name,
        system_prompt=system_prompt,
    )


@Nodes.define(output="chapter_saved")
async def save_chapter(chapter_content: str, completed_chapters: int, title: str, target_directory: str) -> str:
    try:
        current_chapter = completed_chapters + 1
        filename = f"chapter_{current_chapter}.md"
        path = Path(target_directory).resolve() / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(chapter_content)
        logger.info(f"Chapter {current_chapter} saved to {path}")
        return "chapter_saved"
    except Exception as e:
        logger.error(f"Failed to save chapter {current_chapter}: {e}")
        raise


@Nodes.define(output="completed_chapters")
async def update_progress(completed_chapters: int, chapters: List[str], chapter_content: str) -> int:
    logger.info(f"Updating progress, chapter {completed_chapters + 1} ...")
    chapters.append(chapter_content)
    return completed_chapters + 1


@Nodes.define(output="full_course")
async def compile_course(outline: str, chapters: List[str], title: str) -> str:
    full_course = [outline, "\n\n"]
    for idx, chapter in enumerate(chapters, 1):
        full_course.append(f"{chapter}\n\n")
    return "\n".join(full_course)


# File generation functions
def _generate_pdf_sync(full_course: str, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_dir = os.getcwd()
    os.chdir(output_path.parent)

    preamble_path = output_path.parent / "custom_preamble.tex"
    with open(preamble_path, "w", encoding="utf-8") as f:
        f.write(
            r"""
            \usepackage{fontspec}
            \usepackage{emoji}
            \usepackage{float}
            \usepackage{graphicx}
            \setmainfont{DejaVu Serif}
            \setmonofont{DejaVu Sans Mono}
            \floatplacement{figure}{H}
            \floatplacement{table}{H}
            """
        )

    pypandoc.convert_text(
        full_course,
        "pdf",
        format="md",
        outputfile=str(output_path),
        extra_args=[
            "--pdf-engine=lualatex",
            "-V",
            "geometry:margin=1in",
            "-V",
            "documentclass=article",
            "-H",
            str(preamble_path),
            "--highlight-style=kate",
            "--resource-path=.",
            "--no-highlight",
            "--wrap=none",
        ],
    )
    os.remove(preamble_path)
    os.chdir(current_dir)
    return str(output_path)


async def generate_pdf(full_course: str, output_path: Path) -> Optional[str]:
    try:
        result = await anyio.to_thread.run_sync(_generate_pdf_sync, full_course, output_path)
        logger.success(f"Generated PDF at: {result}")
        return result
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


def _generate_docx_sync(full_course: str, output_path: Path) -> str:
    pypandoc.convert_text(
        full_course,
        "docx",
        format="md",
        outputfile=str(output_path),
        extra_args=["--resource-path", str(output_path.parent)],
    )
    return str(output_path)


async def generate_docx(full_course: str, output_path: Path) -> Optional[str]:
    try:
        result = await anyio.to_thread.run_sync(_generate_docx_sync, full_course, output_path)
        logger.success(f"Generated DOCX at: {result}")
        return result
    except Exception as e:
        logger.error(f"DOCX generation failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


def _generate_epub_sync(full_course: str, output_path: Path) -> str:
    pypandoc.convert_text(
        full_course,
        "epub",
        format="md",
        outputfile=str(output_path),
        extra_args=["--highlight-style=kate"],
    )
    return str(output_path)


async def generate_epub(full_course: str, output_path: Path) -> Optional[str]:
    try:
        result = await anyio.to_thread.run_sync(_generate_epub_sync, full_course, output_path)
        logger.success(f"Generated EPUB at: {result}")
        return result
    except Exception as e:
        logger.error(f"EPUB generation failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


def clean_title_for_filename(title: str, max_length: int = 50) -> str:
    """Clean title for use in filenames."""
    title = title.split("\n")[0].strip()[:max_length]
    cleaned = "".join(c if c.isalnum() else "_" for c in title)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_").lower()


@Nodes.define(output="full_course_saved")
async def save_full_course(
    full_course: str,
    title: str,
    target_directory: str,
    pdf_generation: bool,
    docx_generation: bool,
    epub_generation: bool,
) -> Dict[str, Optional[str]]:
    output_dir = Path(target_directory).resolve()
    clean_filename = clean_title_for_filename(title)
    markdown_path = (output_dir / f"{clean_filename}.md").resolve()

    # Process Mermaid diagrams
    from ai_course_generator.mermaid_processor import MermaidProcessor

    mermaid_processor = MermaidProcessor(target_directory=target_directory, filename_prefix=clean_filename)
    processed_content = mermaid_processor.process_content(full_course)

    # Save processed markdown
    async with aiofiles.open(markdown_path, "w", encoding="utf-8") as f:
        await f.write(processed_content)

    generated_files = {"md": str(markdown_path)}
    if pdf_generation:
        pdf_path = (output_dir / f"{clean_filename}.pdf").resolve()
        generated_files["pdf"] = await generate_pdf(processed_content, pdf_path)
    if docx_generation:
        docx_path = (output_dir / f"{clean_filename}.docx").resolve()
        generated_files["docx"] = await generate_docx(processed_content, docx_path)
    if epub_generation:
        epub_path = (output_dir / f"{clean_filename}.epub").resolve()
        generated_files["epub"] = await generate_epub(processed_content, epub_path)

    return generated_files


@Nodes.define(output=None)
async def end() -> None:
    logger.info("Course generation completed.")


# Workflow setup
async def load_system_prompt(subject: str, level: str) -> str:
    env = Environment(loader=FileSystemLoader("prompts"))
    template = env.get_template("course_system_prompt.j2")
    return template.render(subject=subject, level=level)


async def create_workflow(request: CourseRequest) -> Workflow:
    system_prompt = await load_system_prompt(request.subject, request.level)

    # Update LLM nodes with dynamic system prompt
    Nodes.llm_node(
        model=request.model_name,
        system_prompt=system_prompt,
        output="title",
        prompt_file="prompts/course_title_generation_prompt.j2",
        max_tokens=100,
    )(generate_title)

    Nodes.llm_node(
        model=request.model_name,
        system_prompt=system_prompt,
        output="outline",
        prompt_file="prompts/course_outline_prompt.j2",
        **OUTLINE_PARAMS,
    )(generate_outline)

    # Pass system_prompt to generate_chapter
    workflow = Workflow("validate_input")
    workflow.sequence("validate_input", "generate_title", "generate_outline", "save_outline")
    workflow.then("chapter_loop")
    workflow.then("generate_chapter", condition=lambda ctx: ctx["loop_status"] == "next")
    workflow.node("generate_chapter").node_inputs["generate_chapter"].append("system_prompt")
    workflow.then("save_chapter")
    workflow.then("update_progress")
    workflow.then("chapter_loop")
    workflow.then("compile_course", condition=lambda ctx: ctx["loop_status"] == "complete")
    workflow.then("save_full_course")
    workflow.then("end")

    # Add the content streaming observer
    workflow.add_observer(content_stream_observer)

    # Inject system_prompt into initial context
    workflow.context = {"system_prompt": system_prompt}

    return workflow


async def generate_course(request: CourseRequest) -> Dict[str, Optional[str]]:
    workflow = await create_workflow(request)
    engine = workflow.build()
    initial_context = {
        **request.model_dump(),
        "completed_chapters": 0,
        "chapters": [],
        "system_prompt": workflow.context["system_prompt"],
    }
    final_context = await engine.run(initial_context)

    if "full_course_saved" not in final_context:
        raise ValueError("Course generation failed")
    return final_context["full_course_saved"]


# Main execution
async def main():
    check_dependencies()
    request = CourseRequest(
        subject="Python Programming",
        number_of_chapters=5,
        level="intermediate",
        words_by_chapter=1000,
        target_directory="./courses/python3",
        pdf_generation=True,
        docx_generation=True,
        epub_generation=False,
        model_name="gemini/gemini-2.0-flash",
    )

    output_paths = await generate_course(request)
    for format_, path in output_paths.items():
        if path:
            logger.success(f"{format_.upper()} version available at: {path}")


if __name__ == "__main__":
    asyncio.run(main())
