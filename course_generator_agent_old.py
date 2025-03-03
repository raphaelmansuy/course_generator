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
#     "jinja2>=3.0"
# ]
# ///

import anyio
import aiofiles
from pathlib import Path
from typing import List, Optional
from loguru import logger
from litellm import acompletion
from qflow import Workflow, WorkflowEngine
from pydantic import BaseModel
from mermaid_processor import MermaidProcessor
import jinja2

MODEL = "gemini/gemini-2.0-flash"


def load_template(template_path: str) -> jinja2.Template:
    """Load a Jinja2 template from a file."""
    template_loader = jinja2.FileSystemLoader(searchpath="./prompts")
    template_env = jinja2.Environment(loader=template_loader)
    return template_env.get_template(template_path)


def calculate_max_tokens(words: int, is_outline: bool = False) -> int:
    """Calculate max tokens needed based on word count.
    For outlines, we need extra tokens for structure and metadata.
    For chapters, we use a higher token-to-word ratio to account for code blocks and formatting.
    Adds a 50% buffer to prevent truncation."""
    if is_outline:
        # For outlines, we need more tokens to account for structure
        base_tokens = 5000  # Base tokens for structure
        per_chapter_tokens = 500  # Tokens per chapter in outline
        return base_tokens + (
            per_chapter_tokens * 5
        )  # Assuming max 5 chapters for safety
    else:
        # For chapters, use a higher ratio (4 tokens per word) plus 100% buffer
        return int(words * 4 * 2.0)


# Default parameters for different content types
DEFAULT_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 4000,  # Higher default for safety
    "top_p": 0.95,  # Maintain good quality while allowing some creativity
    "frequency_penalty": 0.2,  # Reduce repetition
}

OUTLINE_PARAMS = {
    **DEFAULT_PARAMS,
    "temperature": 0.6,  # More focused for structure
    "max_tokens": 3500,  # Increased for detailed outlines
}

CHAPTER_PARAMS = {
    **DEFAULT_PARAMS,
    "temperature": 0.7,  # Balance between creativity and focus
    "max_tokens": 4000,  # Will be overridden based on words_by_chapter
}


class CourseRequest(BaseModel):
    subject: str
    number_of_chapters: int
    level: str
    words_by_chapter: int
    target_directory: str
    pdf_generation: bool = True  # New flag with default False
    docx_generation: bool = True  # New flag with default True
    epub_generation: bool = False  # New flag for EPUB generation
    model_name: str = MODEL  # Add model_name with default from MODEL constant


class CourseGenerationRequest(CourseRequest):
    system_prompt_outline: str
    system_prompt_chapter: str


def create_workflow() -> Workflow:
    wf = Workflow(entry_node="validate_input", version="1.0.0")

    # Nodes
    wf.add_node(
        name="validate_input",
        func=validate_input,
        inputs=["subject", "number_of_chapters", "level", "words_by_chapter"],
        output="validation_result",
        description="Validate input parameters",
    )

    wf.add_node(
        name="generate_title",
        func=generate_title,
        inputs=["subject", "level", "model_name"],
        output="title",
        description="Generate course title",
    )

    wf.add_node(
        name="generate_outline",
        func=generate_outline,
        inputs=["title", "system_prompt_outline", "number_of_chapters", "level", "model_name"],
        output="outline",
        description="Generate course outline using system prompt",
    )

    wf.add_node(
        name="save_outline",
        func=save_outline,
        inputs=["outline", "title", "target_directory"],
        output="outline_saved",
        description="Save outline to file",
    )

    wf.add_node(
        name="chapter_loop",
        func=chapter_loop,
        inputs=["number_of_chapters", "completed_chapters"],
        output="loop_status",
        description="Manage chapter generation loop",
    )

    wf.add_node(
        name="generate_chapter",
        func=generate_chapter,
        inputs=[
            "title",
            "outline",
            "number_of_chapters",
            "words_by_chapter",
            "system_prompt_chapter",
            "level",
            "completed_chapters",
            "model_name"
        ],
        output="chapter_content",
        description="Generate a chapter using system prompt",
    )

    wf.add_node(
        name="save_chapter",
        func=save_chapter,
        inputs=["chapter_content", "completed_chapters", "title", "target_directory"],
        output="chapter_saved",
        description="Save chapter to file",
    )

    wf.add_node(
        name="update_progress",
        func=update_progress,
        inputs=["completed_chapters", "chapters", "chapter_content"],
        output="completed_chapters",
        description="Update progress with new chapter",
    )

    wf.add_node(
        name="compile_course",
        func=compile_course,
        inputs=["outline", "chapters", "title"],
        output="full_course",
        description="Compile full course markdown",
    )

    wf.add_node(
        name="save_full_course",
        func=save_full_course,
        inputs=["full_course", "title", "target_directory", "pdf_generation", "docx_generation", "epub_generation"],
        output="full_course_saved",
        description="Save full course to file",
    )

    wf.add_node(name="end", func=end, inputs=[], output="", description="End node")

    # Transitions
    wf.add_transition("validate_input", "success", "generate_title")
    wf.add_transition("generate_title", "success", "generate_outline")
    wf.add_transition("generate_outline", "success", "save_outline")
    wf.add_transition("save_outline", "success", "chapter_loop")
    wf.add_transition("chapter_loop", "next", "generate_chapter")
    wf.add_transition("generate_chapter", "success", "save_chapter")
    wf.add_transition("save_chapter", "success", "update_progress")
    wf.add_transition("update_progress", "success", "chapter_loop")
    wf.add_transition("chapter_loop", "complete", "compile_course")
    wf.add_transition("compile_course", "success", "save_full_course")
    wf.add_transition("save_full_course", "success", "end")

    return wf


async def generate_content(
    prompt: str,
    system_prompt: Optional[str] = None,
    words: Optional[int] = None,
    model_name: str = MODEL,  # Add model_name parameter with default from MODEL
    **kwargs,
):
    """Generate content using acompletion with optional system prompt and token calculation."""
    # Calculate max tokens if words are provided
    max_tokens = kwargs.get('max_tokens')
    if words and not max_tokens:
        max_tokens = calculate_max_tokens(words)
    
    # Prepare messages with optional system prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Prepare parameters for acompletion
    params = {
        **DEFAULT_PARAMS,
        "model": model_name,
        "messages": messages,
        **({"max_tokens": max_tokens} if max_tokens else {}),
        **kwargs
    }
    
    try:
        response = await acompletion(**params)
        # Extract clean markdown content
        content = response.choices[0].message.content
        return extract_markdown_or_content(content)
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        raise


async def validate_input(
    subject: str, number_of_chapters: int, level: str, words_by_chapter: int
) -> str:
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


async def generate_title(subject: str, level: str, model_name: str = MODEL) -> str:
    """
    Generate a compelling course title based on subject and level.
    
    Args:
        subject (str): The subject of the course
        level (str): The educational level of the course
        model_name (str, optional): The AI model to use. Defaults to MODEL.
    
    Returns:
        str: A generated course title
    """
    template = load_template('course_title_generation_prompt.j2')
    prompt = template.render(subject=subject, level=level)
    
    template_system = load_template('course_title_system_prompt.j2')
    system_prompt = template_system.render(subject=subject, level=level)
    
    return await generate_content(
        prompt=prompt,
        system_prompt=system_prompt,
        model_name=model_name,
        max_tokens=100  # Limit tokens to ensure concise title
    )


async def _generate_outline_implementation(
    title: str, 
    system_prompt_outline: str, 
    number_of_chapters: int, 
    level: str,
    model_name: str = MODEL
) -> str:
    """Generate a course outline with appropriate token limits for outline generation."""
    template = load_template('course_outline_prompt.j2')
    prompt = template.render(
        title=title, 
        level=level, 
        number_of_chapters=number_of_chapters
    )
    
    return await generate_content(
        prompt=prompt,
        system_prompt=system_prompt_outline,
        words=2000,  # Allocate more tokens for detailed outline
        model_name=model_name,
        **OUTLINE_PARAMS
    )


async def generate_outline(
    title: str, 
    system_prompt_outline: str, 
    number_of_chapters: int, 
    level: str,
    model_name: str = MODEL
) -> str:
    """
    Generate course outline with error handling
    """
    try:
        logger.info("Generating course outline...")
        return await _generate_outline_implementation(title, system_prompt_outline, number_of_chapters, level, model_name)
    except Exception as e:
        logger.error(f"Outline generation failed: {str(e)}")
        raise


async def save_outline(outline: str, title: str, target_directory: str) -> str:
    """Save course outline to a markdown file.
    
    Args:
        outline: Course outline content
        title: Course title for logging
        target_directory: Directory to save the outline in
        
    Returns:
        str: 'outline_saved' on success
        
    Raises:
        OSError: If directory creation or file writing fails
    """
    try:
        # Prepare file path
        path = Path(target_directory).resolve() / "outline.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(outline)
            
        logger.success(f"Saved outline for '{title}' to {path}")
        return "outline_saved"
        
    except Exception as e:
        logger.error(f"Failed to save outline for '{title}': {e}")
        raise


async def chapter_loop(number_of_chapters: int, completed_chapters: int) -> str:
    if completed_chapters < number_of_chapters:
        return "next"
    return "complete"


async def generate_chapter(
    title: str,
    outline: str,
    number_of_chapters: int,
    words_by_chapter: int,
    system_prompt_chapter: str,
    level: str,
    completed_chapters: int,
    model_name: str = MODEL
) -> str:
    """Generate a chapter with dynamically calculated token limits based on word count."""
    # Extract the specific chapter details from the outline
    template = load_template('course_chapter_prompt.j2')
    prompt = template.render(
        title=title,
        level=level,
        number_of_chapters=number_of_chapters,
        completed_chapters=completed_chapters,
        outline=outline,
        words_by_chapter=words_by_chapter
    )
    
    return await generate_content(
        prompt=prompt,
        system_prompt=system_prompt_chapter,
        words=words_by_chapter,
        model_name=model_name,
        **CHAPTER_PARAMS
    )


async def save_chapter(
    chapter_content: str, completed_chapters: int, title: str, target_directory: str
) -> str:
    """Save a chapter to a markdown file.
    
    Args:
        chapter_content: The chapter content to save
        completed_chapters: Number of chapters completed so far
        title: Course title for logging
        target_directory: Directory to save the chapter in
        
    Returns:
        str: 'chapter_saved' on success
        
    Raises:
        RuntimeError: If saving fails
    """
    try:
        current_chapter = completed_chapters + 1
        filename = f"chapter_{current_chapter}.md"
        path = Path(target_directory) / filename
        
        # Create directory synchronously - Path.mkdir is not async
        path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(chapter_content)
            
        logger.info(f"Chapter {current_chapter} saved to {path}")
        return "chapter_saved"
        
    except Exception as e:
        logger.error(f"Failed to save chapter {current_chapter}: {e}")
        raise RuntimeError(f"Failed to save chapter {current_chapter}: {e}")


async def update_progress(
    completed_chapters: int, chapters: List[str], chapter_content: str
) -> int:
    logger.info(f"Updating progress, chapter {completed_chapters + 1} ...")
    chapters.append(chapter_content)
    return completed_chapters + 1


async def compile_course(outline: str, chapters: List[str], title: str) -> str:
    full_course = [
        outline,
        "\n\n",
    ]
    for idx, chapter in enumerate(chapters, 1):
        full_course.append(f"{chapter}\n\n")
    return "\n".join(full_course)


async def generate_pdf(full_course: str, output_path: Path) -> Optional[str]:
    """
    Generate a PDF from markdown content using pypandoc.
    
    Args:
        full_course (str): Markdown content to convert
        output_path (Path): Path to save the generated PDF
    
    Returns:
        Optional[str]: Path to generated PDF or None if generation fails
    """
    try:
        import pypandoc
        import os
        import sys
        
        # Log detailed path information
        logger.info(f"PDF Generation Paths:")
        logger.info(f"  Output Path: {output_path}")
        logger.info(f"  Output Directory: {output_path.parent}")
        logger.info(f"  Current Working Directory: {os.getcwd()}")
        
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert markdown to PDF with local image support
        output_filename = str(output_path)
        current_dir = os.getcwd()  # Store current working directory
        
        try:
            # Change to the directory containing the markdown file to resolve local image paths
            os.chdir(output_path.parent)
            
            # Create a temporary LaTeX preamble to handle Unicode and floats
            latex_preamble = r"""
            \usepackage{fontspec}
            \usepackage{emoji}
            \usepackage{float}
            \usepackage{graphicx}
            \setmainfont{DejaVu Serif}
            \setmonofont{DejaVu Sans Mono}
            \floatplacement{figure}{H}
            \floatplacement{table}{H}
            """
            
            # Temporary file for LaTeX preamble
            preamble_path = output_path.parent / 'custom_preamble.tex'
            
            # Ensure parent directory exists
            preamble_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write preamble file
            with open(preamble_path, 'w', encoding='utf-8') as f:
                f.write(latex_preamble)
            
            logger.info(f"Preamble file created at: {preamble_path}")
            
            # Verify preamble file exists
            if not preamble_path.exists():
                raise FileNotFoundError(f"Preamble file not created: {preamble_path}")
            
            # Log content of full_course to help diagnose issues
            logger.debug(f"Full Course Content Length: {len(full_course)} characters")
            logger.debug(f"First 500 characters of course content:\n{full_course[:500]}")
            
            pypandoc.convert_text(
                full_course, 
                'pdf', 
                format='md', 
                outputfile=output_filename,
                extra_args=[
                    '--pdf-engine=lualatex',  # Corrected to lualatex
                    '-V', 'geometry:margin=1in',
                    '-V', 'documentclass=article',
                    '-H', str(preamble_path),  # Include custom preamble
                    '--highlight-style=kate',
                    '--resource-path=.',
                    '--no-highlight',
                    '--wrap=none',  # Prevent line wrapping
                    '--default-image-extension=pdf,png,jpg'
                ]
            )
            
            # Clean up temporary preamble file
            os.remove(preamble_path)
        
        finally:
            # Always change back to the original directory
            os.chdir(current_dir)
        
        logger.success(f"Generated PDF at: {output_path}")
        return str(output_path)
    
    except Exception as e:
        logger.warning(f"PDF generation failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())  # Log full traceback for debugging
        return None


async def generate_docx(full_course: str, output_path: Path) -> Optional[str]:
    """Convert markdown to DOCX using pypandoc with explicit parameter binding"""
    try:
        import pypandoc
        from functools import partial
        from anyio import to_thread

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create parameter-bound conversion function
        conversion_call = partial(
            pypandoc.convert_text,
            source=full_course,
            to='docx',
            format='md',
            outputfile=str(output_path),
            extra_args=['--resource-path', str(output_path.parent)]
        )

        await to_thread.run_sync(conversion_call)
        return str(output_path)
    except Exception as e:
        logger.error(f"DOCX conversion error: {e}")
        return None


async def generate_epub(full_course: str, output_path: Path) -> Optional[str]:
    """
    Generate an EPUB from markdown content using pypandoc.
    
    Args:
        full_course (str): Markdown content to convert
        output_path (Path): Path to save the generated EPUB
    
    Returns:
        Optional[str]: Path to generated EPUB or None if generation fails
    """
    try:
        import pypandoc
        
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert markdown to EPUB
        output_filename = str(output_path)
        pypandoc.convert_text(
            full_course, 
            'epub', 
            format='md', 
            outputfile=output_filename,
            extra_args=[
                '--highlight-style=kate'
            ]
        )
        
        logger.success(f"Generated EPUB at: {output_path}")
        return str(output_path)
    
    except Exception as e:
        logger.warning(f"EPUB generation failed: {e}")
        return None


def clean_title_for_filename(title: str, max_length: int = 50) -> str:
    """Create a clean filename from a title by replacing non-alphanumeric chars with underscores.
    
    Args:
        title: The title to clean
        max_length: Maximum length of the resulting filename (default: 50)
        
    Returns:
        A cleaned filename string no longer than max_length
    """
    # Get first line if multiple lines
    title = title.split('\n')[0].strip()
    
    # Take first N chars to avoid too-long filenames
    truncated = title[:max_length]
    
    # Replace non-alphanumeric with underscores
    cleaned = ''.join([c if c.isalnum() else '_' for c in truncated])
    
    # Remove repeated underscores
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    
    return cleaned.strip('_').lower()


async def save_markdown_content(content: str, filepath: Path) -> None:
    """Save markdown content to a file asynchronously."""
    async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
        await f.write(content)


async def generate_additional_formats(
    content: str,
    output_dir: Path,
    filename: str,
    formats: dict[str, bool]
) -> dict[str, Optional[str]]:
    """
    Generate additional file formats (PDF, DOCX, EPUB) based on markdown content.
    
    Args:
        content: Markdown content to convert
        output_dir: Directory to save files in
        filename: Base filename without extension
        formats: Dict of format flags {'pdf': bool, 'docx': bool, 'epub': bool}
    
    Returns:
        dict: Paths to generated files keyed by format
    """

    # Ensure output directory exists and is clean
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process markdown with Mermaid diagrams
    target_directory = output_dir.as_posix()
    mermaid_processor = MermaidProcessor(target_directory=target_directory, filename_prefix=filename)
    processed_content = mermaid_processor.process_content(content)

    # Save processed markdown
    processed_markdown_path = output_dir / f"{filename}_processed.md"
    await save_markdown_content(processed_content, processed_markdown_path)

    # Generate additional formats using processed markdown
    generated_files = {
        'md': processed_markdown_path
    }

    if formats.get('pdf', False):
        pdf_path = output_dir / f"{filename}.pdf"
        pdf_result = await generate_pdf(processed_content, pdf_path)
        generated_files['pdf'] = pdf_result

    if formats.get('docx', False):
        docx_path = output_dir / f"{filename}.docx"
        docx_result = await generate_docx(processed_content, docx_path)
        generated_files['docx'] = docx_result

    if formats.get('epub', False):
        epub_path = output_dir / f"{filename}.epub"
        epub_result = await generate_epub(processed_content, epub_path)
        generated_files['epub'] = epub_result

    return generated_files


async def save_full_course(
    full_course: str,
    title: str, 
    target_directory: str,
    pdf_generation: bool = False,
    docx_generation: bool = True,
    epub_generation: bool = False
) -> dict[str, Optional[str]]:
    """
    Save course content in markdown and generate additional formats if requested.
    
    Args:
        full_course: Complete course content in markdown
        title: Course title for file naming
        target_directory: Directory to save files in
        pdf_generation: Whether to generate PDF
        docx_generation: Whether to generate DOCX
        epub_generation: Whether to generate EPUB
        
    Returns:
        dict: Paths to generated files {'md': path, 'docx': path, ...}
        
    Raises:
        RuntimeError: If saving fails
    """
    # Clean title for filename
    clean_filename = clean_title_for_filename(title)
    output_dir = Path(target_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the markdown file
    markdown_path = output_dir / f"{clean_filename}.md"
    await save_markdown_content(full_course, markdown_path)


    # Prepare formats dictionary
    formats = {
        'pdf': pdf_generation,
        'docx': docx_generation,
        'epub': epub_generation
    }

    # Use generate_additional_formats with Mermaid processing
    return await generate_additional_formats(
        content=full_course, 
        output_dir=output_dir, 
        filename=clean_filename, 
        formats=formats
    )


def extract_markdown_or_content(content: str) -> str:
    """
    Extract clean markdown content from potentially wrapped markdown blocks.
    
    Args:
        content (str): Raw content potentially containing markdown block delimiters
    
    Returns:
        str: Cleaned markdown content without block delimiters
    """
    if not content:
        return ""
    
    # Split content into lines
    lines = content.strip().split("\n")
    
    # Remove leading ```markdown or ``` block
    if lines and (lines[0].startswith("```markdown") or lines[0].startswith("```")):
        lines = lines[1:]
    
    # Remove trailing ``` block
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    
    # Rejoin and strip the content
    return "\n".join(lines).strip()


async def end() -> None:
    logger.info("Course generation completed.")


async def generate_system_prompt(subject: str, level: str) -> str:
    template = load_template('course_system_prompt.j2')
    return template.render(subject=subject, level=level)


async def generate_course(request: CourseRequest) -> dict:
    """Execute course generation workflow and return output path"""
    wf = create_workflow()
    engine = WorkflowEngine(wf)
    
    system_prompt = await generate_system_prompt(request.subject, request.level)
    
    # Convert to CourseGenerationRequest with system prompts
    generation_request = CourseGenerationRequest(
        **request.model_dump(),
        system_prompt_outline=system_prompt,
        system_prompt_chapter=system_prompt
    )
    
    # Spread request fields into context for workflow compatibility
    initial_context = {
        **generation_request.model_dump(),
        "completed_chapters": 0,
        "chapters": []
    }
    
    final_state = await engine.execute(initial_context)
    
    if 'outline' not in final_state.context:
        raise ValueError("Outline generation failed - missing in context")
    if 'chapters' not in final_state.context:
        raise ValueError("Chapter generation failed - missing in context")
    
    return await save_full_course(
        await compile_course(
            final_state.context["outline"],
            final_state.context["chapters"],
            final_state.context["title"]
        ),
        final_state.context["title"],
        final_state.context["target_directory"],
        pdf_generation=final_state.context["pdf_generation"],
        docx_generation=final_state.context["docx_generation"],
        epub_generation=final_state.context["epub_generation"]
    )


async def main():
    request = CourseRequest(
        subject="Change management for AI adoption",
        number_of_chapters=1,
        level="beginner",
        words_by_chapter=1500,
        target_directory="./courses",
        pdf_generation=True,
        epub_generation=True
    )
    
    output_path = await generate_course(request)
    if output_path['docx']:
        logger.success(f"DOCX version available at: {output_path['docx']}")
    if output_path['pdf']:
        logger.success(f"PDF version available at: {output_path['pdf']}")
    if output_path['epub']:
        logger.success(f"EPUB version available at: {output_path['epub']}")
    logger.success(f"Course generation complete. Output saved to: {output_path['md']}")


if __name__ == "__main__":
    anyio.run(main)
