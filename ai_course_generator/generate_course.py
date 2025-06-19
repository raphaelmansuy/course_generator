#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru",
#     "litellm",
#     "pydantic>=2.0",
#     "anyio",
#     "pypandoc",
#     "typer",
#     "rich",
#     "aiofiles",
#     "quantalogic-flow"
# ]
# ///

import anyio
import typer
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt

from ai_course_generator.course_generator_agent import CourseRequest, generate_course

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def generate(
    subject: str = typer.Option(None, "--subject", "-s", help="Course subject"),
    number_of_chapters: int = typer.Option(None, "--number-of-chapters", "-n", help="Number of chapters"),
    level: str = typer.Option(None, "--level", "-l", help="Difficulty level (beginner/intermediate/advanced)"),
    words_by_chapter: int = typer.Option(None, "--words-by-chapter", "-w", help="Target word count per chapter"),
    target_directory: str = typer.Option(None, "--target-directory", "-o", help="Output directory path"),
    pdf_generation: bool = typer.Option(True, "--pdf/--no-pdf", help="Generate PDF output"),
    docx_generation: bool = typer.Option(True, "--docx/--no-docx", help="Generate DOCX output"),
    epub_generation: bool = typer.Option(False, "--epub/--no-epub", help="Generate EPUB output"),
    model_name: str = typer.Option("gemini/gemini-2.0-flash", "--model-name", "-m", help="AI model to use"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
):
    """Generate a course with the given parameters."""

    # Check if any required parameter is missing and switch to interactive mode
    if not interactive and any(
        param is None for param in [subject, number_of_chapters, level, words_by_chapter, target_directory]
    ):
        interactive = True
        console.print("[yellow]Missing required parameters, switching to interactive mode...[/yellow]")

    if interactive:
        console.print("\n[bold]Course Generator Interactive Mode[/bold]\n")
        subject = Prompt.ask("Course Subject", default=subject or "Python Programming")
        number_of_chapters = IntPrompt.ask("Number of Chapters", default=number_of_chapters or 5)
        level = Prompt.ask(
            "Difficulty Level", choices=["beginner", "intermediate", "advanced"], default=level or "intermediate"
        )
        words_by_chapter = IntPrompt.ask("Words per Chapter", default=words_by_chapter or 1000)
        target_directory = Prompt.ask("Output Directory", default=target_directory or "./courses/python3")

        pdf_generation = Confirm.ask("Generate PDF version?", default=pdf_generation)
        docx_generation = Confirm.ask("Generate DOCX version?", default=docx_generation)
        epub_generation = Confirm.ask("Generate EPUB version?", default=epub_generation)

        model_name = Prompt.ask("AI Model to use for generation", default=model_name or "gemini/gemini-2.0-flash")

        console.print("\n[bold]Summary of your course:[/bold]")
        console.print(f"  Subject: {subject}")
        console.print(f"  Chapters: {number_of_chapters}")
        console.print(f"  Level: {level}")
        console.print(f"  Words per chapter: {words_by_chapter}")
        console.print(
            f"  Output formats: {'PDF' if pdf_generation else ''} {'DOCX' if docx_generation else ''} {'EPUB' if epub_generation else ''}".strip()
        )
        console.print(f"  Target directory: {target_directory}")

        if not Confirm.ask("\n[bold]Proceed with course generation?[/bold]", default=True):
            raise typer.Abort()

    course_request = CourseRequest(
        subject=subject,
        number_of_chapters=number_of_chapters,
        level=level,
        words_by_chapter=words_by_chapter,
        target_directory=target_directory,
        pdf_generation=pdf_generation,
        docx_generation=docx_generation,
        epub_generation=epub_generation,
        model_name=model_name,
    )

    with console.status("[bold green]Generating course..."):
        result = anyio.run(generate_course, course_request)

    console.print(f"[bold green]\nGenerated Course Plan at: {result}[/bold green]")


if __name__ == "__main__":
    app()
