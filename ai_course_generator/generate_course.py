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
#     "quantalogic"
# ]
# ///

import typer
import anyio
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from ai_course_generator.course_generator_agent import CourseRequest, generate_course

app = typer.Typer(add_completion=False)
console = Console()

@app.command()
def generate(
    subject: str = typer.Option(None, help="Course subject"),
    number_of_chapters: int = typer.Option(None, help="Number of chapters"),
    level: str = typer.Option(None, help="Difficulty level (beginner/intermediate/advanced)"),
    words_by_chapter: int = typer.Option(None, help="Number of words per chapter"),
    target_directory: str = typer.Option(None, help="Target directory for course output"),
    pdf_generation: bool = typer.Option(True, help="Generate PDF version of the course"),
    docx_generation: bool = typer.Option(True, help="Generate DOCX version of the course"),
    epub_generation: bool = typer.Option(False, help="Generate EPUB version of the course"),
    model_name: str = typer.Option("gemini/gemini-2.0-flash", help="AI model to use for generation"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Run in interactive mode")
):
    """Generate a course with the given parameters."""
    # If no required parameters are provided, force interactive mode
    if all(param is None for param in [subject, number_of_chapters, level, words_by_chapter, target_directory]):
        interactive = True
        
    if interactive:
        console.print("[bold green]\nCourse Generator Interactive Mode\n[/bold green]")
        
        # Get required parameters interactively
        subject = Prompt.ask("\n[bold]Course Subject[/bold] (e.g. Python Programming)", default=subject or "Python Programming")
        number_of_chapters = IntPrompt.ask(
            "\n[bold]Number of Chapters[/bold]", 
            default=number_of_chapters or 5,
            show_default=True
        )
        level = Prompt.ask(
            "\n[bold]Difficulty Level[/bold] (beginner/intermediate/advanced)",
            choices=["beginner", "intermediate", "advanced"],
            default=level or "intermediate"
        )
        words_by_chapter = IntPrompt.ask(
            "\n[bold]Words per Chapter[/bold]",
            default=words_by_chapter or 1000,
            show_default=True
        )
        target_directory = Prompt.ask(
            "\n[bold]Target Directory[/bold] for course files",
            default=target_directory or "./courses/python3"
        )
        
        # Get optional parameters
        pdf_generation = Confirm.ask(
            "\n[bold]Generate PDF version[/bold]?",
            default=pdf_generation
        )
        docx_generation = Confirm.ask(
            "\n[bold]Generate DOCX version[/bold]?",
            default=docx_generation
        )
        epub_generation = Confirm.ask(
            "\n[bold]Generate EPUB version[/bold]?",
            default=epub_generation
        )
        
        model_name = Prompt.ask(
            "\n[bold]AI Model[/bold] to use for generation",
            default=model_name or "gemini/gemini-2.0-flash"
        )
        
        console.print("\n[bold]Summary of your course:[/bold]")
        console.print(f"  Subject: {subject}")
        console.print(f"  Chapters: {number_of_chapters}")
        console.print(f"  Level: {level}")
        console.print(f"  Words per chapter: {words_by_chapter}")
        console.print(f"  Output formats: {'PDF' if pdf_generation else ''} {'DOCX' if docx_generation else ''} {'EPUB' if epub_generation else ''}".strip())
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
        model_name=model_name
    )
    
    with console.status("[bold green]Generating course..."):
        result = anyio.run(generate_course, course_request)
    
    console.print(f"[bold green]\nGenerated Course Plan at: {result}[/bold green]")

if __name__ == "__main__":
    app()
