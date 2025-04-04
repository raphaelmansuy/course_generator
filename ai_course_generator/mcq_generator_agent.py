#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru",
#     "litellm==1.61.0",
#     "pydantic>=2.0",
#     "anyio",
#     "quantalogic>=0.35",
#     "jinja2",
#     "typer>=0.9.0",
#     "rich"
# ]
# ///

import asyncio
import json
import os
import random
from typing import List, Optional

import typer
from loguru import logger
from pydantic import BaseModel, Field
from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent
from rich.console import Console
from rich.progress import Progress, TaskID

# Initialize Rich console for colored output
console = Console()

# === Pydantic Models ===
class MCQRequest(BaseModel):
    """Input model for the MCQ generator."""
    topic: str = Field(..., description="The topic for the MCQs")
    difficulty: str = Field(..., description="Difficulty level: beginner, intermediate, advanced")
    num_questions: int = Field(..., description="Number of questions to generate")
    num_options: int = Field(4, description="Number of options per question")
    target_directory: str = Field(..., description="Directory to save the output files")
    output_formats: List[str] = Field(default=["json"], description="Formats to save the MCQs (e.g., json, csv, md)")
    model_name: str = Field("gemini/gemini-2.0-flash", description="LLM model to use")
    batch_size: int = Field(5, description="Batch size (currently unused, set for future optimization)")

class KeyConcepts(BaseModel):
    """Model for key concepts extracted from a topic."""
    concepts: List[str] = Field(..., description="List of key concepts related to the topic")

class QuestionWithAnswer(BaseModel):
    """Intermediate model for a question and its correct answer."""
    question: str
    correct_answer: str

class Distractors(BaseModel):
    """Intermediate model for distractor options."""
    distractors: List[str]

class MCQItem(BaseModel):
    """Model for a complete MCQ item."""
    question: str
    options: List[str]
    correct_answer: int  # 1-based index
    explanation: str = ""  # Populated with detailed explanation
    key_concept: str = Field("", description="The key concept this question relates to")

class Explanation(BaseModel):
    """Model for the explanation of an MCQ item."""
    explanation: str

# === Workflow Nodes ===
@Nodes.define(output="request")
async def validate_input(request: MCQRequest) -> MCQRequest:
    """Validate the MCQRequest parameters."""
    logger.info(f"Validating input: {request}")
    if request.num_questions <= 0:
        raise ValueError("Number of questions must be positive")
    if request.difficulty not in ["beginner", "intermediate", "advanced"]:
        raise ValueError("Invalid difficulty level")
    if request.num_options < 2:
        raise ValueError("Number of options must be at least 2")
    if not request.output_formats:
        raise ValueError("At least one output format must be specified")
    return request

@Nodes.structured_llm_node(
    system_prompt="You are an AI assistant tasked with identifying key concepts for educational content.",
    output="key_concepts",
    response_model=KeyConcepts,
    prompt_template="""
Given the topic '{{ topic }}', identify 5-10 key concepts that are essential for understanding it at the {{ difficulty }} level. Provide only the list of concepts, no explanations.
""",
    max_tokens=500
)
async def extract_key_concepts(topic: str, difficulty: str, model: str) -> KeyConcepts:
    """Extract key concepts from the topic using an LLM."""
    pass  # Implementation handled by the decorator

@Nodes.define(output=None)
async def initialize_mcq_generation(request: MCQRequest, key_concepts: KeyConcepts) -> dict:
    """Initialize the context for MCQ generation with key concepts."""
    logger.info("Initializing MCQ generation")
    num_questions = request.num_questions
    concepts = key_concepts.concepts
    # Distribute questions evenly across concepts, remainder goes to earlier concepts
    questions_per_concept = [num_questions // len(concepts)] * len(concepts)
    for i in range(num_questions % len(concepts)):
        questions_per_concept[i] += 1
    return {
        "topic": request.topic,
        "difficulty": request.difficulty,
        "num_questions": num_questions,
        "num_options": request.num_options,
        "model": request.model_name,
        "current_index": 0,
        "mcq_items": [],
        "output_formats": request.output_formats,
        "target_directory": request.target_directory,
        "key_concepts": concepts,
        "questions_per_concept": questions_per_concept,
        "current_concept_index": 0,
        "progress_task": None  # Placeholder for progress bar task ID
    }

@Nodes.structured_llm_node(
    system_prompt="You are an AI assistant tasked with generating educational multiple-choice questions.",
    output="question_with_answer",
    response_model=QuestionWithAnswer,
    prompt_template="""
Generate a multiple-choice question on the specific concept '{{ current_concept }}' within the topic '{{ topic }}' at the {{ difficulty }} level. Provide the question and the correct answer.
""",
    max_tokens=1000
)
async def generate_question(topic: str, difficulty: str, model: str, current_concept: str) -> QuestionWithAnswer:
    """Generate a question and its correct answer for a specific concept using an LLM."""
    pass  # Implementation handled by the decorator

@Nodes.structured_llm_node(
    system_prompt="Generate plausible distractors for a multiple-choice question.",
    output="distractors",
    response_model=Distractors,
    prompt_template="""
For the question: '{{ question }}' with correct answer: '{{ correct_answer }}', generate {{ num_distractors }} plausible distractors related to '{{ topic }}'.
""",
    max_tokens=1000
)
async def generate_distractors(question: str, correct_answer: str, num_distractors: int, topic: str, model: str) -> Distractors:
    """Generate distractors for the question using an LLM."""
    pass  # Implementation handled by the decorator

@Nodes.define(output="mcq_item")
async def create_mcq_item(question_with_answer: QuestionWithAnswer, distractors: Distractors, current_concept: str) -> MCQItem:
    """Create an MCQ item by combining the question, correct answer, distractors, and key concept."""
    logger.debug(f"Creating MCQ item for question: {question_with_answer.question}")
    options = [question_with_answer.correct_answer] + distractors.distractors
    random.shuffle(options)
    correct_index = options.index(question_with_answer.correct_answer) + 1  # 1-based index
    return MCQItem(
        question=question_with_answer.question,
        options=options,
        correct_answer=correct_index,
        key_concept=current_concept
    )

@Nodes.define(output="mcq_item")
async def validate_mcq_item(mcq_item: MCQItem) -> MCQItem:
    """Validate the MCQ item for uniqueness and correctness."""
    logger.debug(f"Validating MCQ item: {mcq_item.question}")
    if len(set(mcq_item.options)) != len(mcq_item.options):
        raise ValueError(f"Duplicate options in MCQ item: {mcq_item.question}")
    if mcq_item.correct_answer < 1 or mcq_item.correct_answer > len(mcq_item.options):
        raise ValueError(f"Invalid correct answer index for: {mcq_item.question}")
    return mcq_item

@Nodes.define(output=None)
async def prepare_explanation_inputs(mcq_item: MCQItem) -> dict:
    """Prepare inputs for generating the explanation."""
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']  # Support up to 10 options
    correct_index = mcq_item.correct_answer - 1  # Convert to 0-based index
    correct_letter = letters[correct_index]
    correct_option = mcq_item.options[correct_index]
    formatted_options = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(mcq_item.options)])
    logger.debug(f"Prepared explanation inputs for '{mcq_item.question}': correct_letter={correct_letter}, correct_option={correct_option}")
    return {
        "formatted_options": formatted_options,
        "correct_letter": correct_letter,
        "correct_option": correct_option
    }

@Nodes.structured_llm_node(
    system_prompt="You are an AI assistant tasked with providing detailed explanations for multiple-choice questions.",
    output="explanation",
    response_model=Explanation,
    prompt_template="""
Here is a multiple-choice question: '{{ question }}'

Options:
{{ formatted_options }}

The correct answer is option {{ correct_letter }}, which is '{{ correct_option }}'.

Provide a detailed explanation (at least 100 words) for why this is the correct answer and why each of the other options is incorrect. Use the actual option letters (e.g., A, B, C) and their text in your explanation. Do not include placeholders like '{correct_letter}' or '{explanation}' in your response—use the provided values directly.
""",
    max_tokens=1000
)
async def generate_explanation(question: str, formatted_options: str, correct_letter: str, correct_option: str, model: str) -> Explanation:
    """Generate an explanation for the MCQ using an LLM."""
    # Log the rendered prompt for debugging
    rendered_prompt = f"""
Here is a multiple-choice question: '{question}'

Options:
{formatted_options}

The correct answer is option {correct_letter}, which is '{correct_option}'.

Provide a detailed explanation (at least 100 words) for why this is the correct answer and why each of the other options is incorrect. Use the actual option letters (e.g., A, B, C) and their text in your explanation. Do not include placeholders like '{{correct_letter}}' or '{{explanation}}' in your response—use the provided values directly.
"""
    logger.debug(f"Rendered prompt for explanation generation:\n{rendered_prompt}")
    pass  # Implementation handled by the decorator

@Nodes.define(output="mcq_item")
async def set_explanation(mcq_item: MCQItem, explanation: Explanation) -> MCQItem:
    """Set the explanation in the MCQ item."""
    logger.debug(f"Setting explanation for '{mcq_item.question}': {explanation.explanation[:50]}...")
    mcq_item.explanation = explanation.explanation
    return mcq_item

@Nodes.define(output="mcq_items")
async def append_mcq_item(mcq_item: MCQItem, mcq_items: List[MCQItem]) -> List[MCQItem]:
    """Append the MCQ item to the list."""
    logger.debug(f"Appending MCQ item: {mcq_item.question}")
    return mcq_items + [mcq_item]

@Nodes.define(output="current_index")
async def increment_index(current_index: int) -> int:
    """Increment the current question index."""
    logger.debug(f"Incrementing index to {current_index + 1}")
    return current_index + 1

@Nodes.define(output="saved_mcqs")
async def save_mcqs(mcq_items: List[MCQItem], output_formats: List[str], target_directory: str) -> None:
    """Save the MCQ items to files in the specified formats."""
    logger.info(f"Saving {len(mcq_items)} MCQs to {target_directory}")
    os.makedirs(target_directory, exist_ok=True)
    for fmt in output_formats:
        file_path = os.path.join(target_directory, f"mcqs.{fmt}")
        if fmt == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([item.dict() for item in mcq_items], f, indent=2)
        elif fmt == "csv":
            # TODO: Implement CSV saving logic
            logger.warning(f"CSV format not implemented yet: {file_path}")
        elif fmt == "md":
            # TODO: Implement Markdown saving logic
            logger.warning(f"Markdown format not implemented yet: {file_path}")
        else:
            logger.warning(f"Unsupported format: {fmt}")

# === Workflow Construction ===
def create_mcq_workflow() -> Workflow:
    """Construct the workflow for generating MCQs with explanations."""
    wf = Workflow("validate_input")
    
    # Define the sequence of nodes
    wf.node("validate_input").then("extract_key_concepts")
    wf.node("extract_key_concepts").then("initialize_mcq_generation")
    wf.node("initialize_mcq_generation").then("generate_question")
    wf.node("generate_question").then("generate_distractors")
    wf.node("generate_distractors").then("create_mcq_item")
    wf.node("create_mcq_item").then("validate_mcq_item")
    wf.node("validate_mcq_item").then("prepare_explanation_inputs")
    wf.node("prepare_explanation_inputs").then("generate_explanation")
    wf.node("generate_explanation").then("set_explanation")
    wf.node("set_explanation").then("append_mcq_item")
    wf.node("append_mcq_item").then("increment_index")
    wf.node("increment_index").then("save_mcqs")
    
    # Define the loop transition with concept switching
    wf.transitions["increment_index"] = [
        ("generate_question", lambda ctx: (
            ctx["current_index"] < ctx["num_questions"] and
            sum(ctx["questions_per_concept"][:ctx["current_concept_index"] + 1]) > ctx["current_index"]
        )),
        ("generate_question", lambda ctx: (
            ctx["current_index"] < ctx["num_questions"] and
            sum(ctx["questions_per_concept"][:ctx["current_concept_index"] + 1]) <= ctx["current_index"] and
            ctx["current_concept_index"] + 1 < len(ctx["key_concepts"]) and
            (ctx.__setitem__("current_concept_index", ctx["current_concept_index"] + 1) or True)
        )),
        ("save_mcqs", lambda ctx: ctx["current_index"] >= ctx["num_questions"])
    ]
    
    # Input mappings for each node
    wf.node_input_mappings["validate_input"] = {"request": "request"}
    wf.node_input_mappings["extract_key_concepts"] = {
        "topic": lambda ctx: ctx["request"].topic,
        "difficulty": lambda ctx: ctx["request"].difficulty,
        "model": lambda ctx: ctx["request"].model_name
    }
    wf.node_input_mappings["initialize_mcq_generation"] = {
        "request": "request",
        "key_concepts": "key_concepts"
    }
    wf.node_input_mappings["generate_question"] = {
        "topic": "topic",
        "difficulty": "difficulty",
        "model": "model",
        "current_concept": lambda ctx: ctx["key_concepts"][ctx["current_concept_index"]]
    }
    wf.node_input_mappings["generate_distractors"] = {
        "question": lambda ctx: ctx["question_with_answer"].question,
        "correct_answer": lambda ctx: ctx["question_with_answer"].correct_answer,
        "num_distractors": lambda ctx: ctx["num_options"] - 1,
        "topic": "topic",
        "model": "model"
    }
    wf.node_input_mappings["create_mcq_item"] = {
        "question_with_answer": "question_with_answer",
        "distractors": "distractors",
        "current_concept": lambda ctx: ctx["key_concepts"][ctx["current_concept_index"]]
    }
    wf.node_input_mappings["validate_mcq_item"] = {"mcq_item": "mcq_item"}
    wf.node_input_mappings["prepare_explanation_inputs"] = {"mcq_item": "mcq_item"}
    wf.node_input_mappings["generate_explanation"] = {
        "question": lambda ctx: ctx["mcq_item"].question,
        "formatted_options": "formatted_options",
        "correct_letter": "correct_letter",
        "correct_option": "correct_option",
        "model": "model"
    }
    wf.node_input_mappings["set_explanation"] = {
        "mcq_item": "mcq_item",
        "explanation": "explanation"
    }
    wf.node_input_mappings["append_mcq_item"] = {
        "mcq_item": "mcq_item",
        "mcq_items": "mcq_items"
    }
    wf.node_input_mappings["increment_index"] = {"current_index": "current_index"}
    wf.node_input_mappings["save_mcqs"] = {
        "mcq_items": "mcq_items",
        "output_formats": "output_formats",
        "target_directory": "target_directory"
    }
    
    logger.debug("Workflow nodes registered: %s", wf._nodes.keys() if hasattr(wf, '_nodes') else "Unknown")
    logger.info("MCQ workflow created")
    return wf

# === Observer for Progress Tracking ===
async def mcq_stream_observer(event: WorkflowEvent):
    """Observer to track and display workflow progress with a progress bar and colored output."""
    total = event.context.get("num_questions", 0)
    index = event.context.get("current_index", 0)
    
    if event.node_name == "initialize_mcq_generation":
        # Initialize progress bar when workflow starts
        with Progress() as progress:
            task = progress.add_task("[yellow]Generating MCQs", total=total)
            event.context["progress"] = progress  # Store progress object
            event.context["progress_task"] = task  # Store task ID
    
    elif event.node_name == "generate_question":
        concept = event.context["key_concepts"][event.context["current_concept_index"]]
        progress = event.context.get("progress")
        task = event.context.get("progress_task")
        if progress and task:
            progress.update(task, completed=index + 1, description=f"[yellow]Generating for '{concept}'")
    
    elif event.node_name == "generate_explanation":
        console.print(f"[cyan]Generating explanation for question {index + 1}/{total}")
    
    elif event.node_name == "save_mcqs":
        console.print(f"[green]Saving MCQs to {event.context['target_directory']}")

# === Workflow Execution ===
async def generate_mcqs(request: MCQRequest):
    """Run the MCQ generation workflow."""
    logger.info(f"Starting MCQ generation for topic: {request.topic}")
    workflow = create_mcq_workflow()
    engine = workflow.build()
    engine.add_observer(mcq_stream_observer)  # Attach observer for progress tracking
    initial_context = {"request": request}
    result = await engine.run(initial_context)
    logger.info("MCQ generation completed")
    console.print(f"[green]MCQs successfully generated and saved to {request.target_directory}")
    return result

# === Typer CLI Integration ===
app = typer.Typer(
    help="Generate multiple-choice questions based on a topic and difficulty level"
)

@app.command()
def generate(
    topic: Optional[str] = typer.Argument(None, help="Topic for the MCQs"),
    difficulty: Optional[str] = typer.Argument(None, help="Difficulty: beginner, intermediate, advanced"),
    num_questions: Optional[int] = typer.Argument(None, help="Number of questions to generate"),
    num_options: int = typer.Option(4, help="Number of options per question"),
    target_directory: str = typer.Option("./mcqs", help="Directory to save output files"),
    output_formats: List[str] = typer.Option(["json"], help="Output formats (e.g., json, csv, md)"),
    model_name: str = typer.Option("gemini/gemini-2.0-flash", help="LLM model name"),
    batch_size: int = typer.Option(5, help="Batch size (for future use)")
):
    """Generate MCQs and save them to the specified directory. Enters interactive mode if parameters are omitted."""
    try:
        # Check if required arguments are provided; if not, enter interactive mode
        interactive = topic is None or difficulty is None or num_questions is None

        if interactive:
            console.print("[bold cyan]Entering interactive mode...[/bold cyan]")
            topic = typer.prompt("Enter the topic for the MCQs")
            difficulty = typer.prompt(
                "Enter the difficulty level (beginner, intermediate, advanced)",
                type=str,
                default="beginner",
                show_default=True
            ).lower()
            while difficulty not in ["beginner", "intermediate", "advanced"]:
                console.print("[red]Invalid difficulty level. Please choose: beginner, intermediate, advanced[/red]")
                difficulty = typer.prompt("Enter the difficulty level").lower()
            num_questions = typer.prompt(
                "Enter the number of questions to generate",
                type=int,
                default=5,
                show_default=True
            )
            num_options = typer.prompt(
                "Enter the number of options per question",
                type=int,
                default=num_options,
                show_default=True
            )
            target_directory = typer.prompt(
                "Enter the target directory for output files",
                default=target_directory,
                show_default=True
            )
            output_formats_str = typer.prompt(
                "Enter output formats (comma-separated, e.g., json, csv, md)",
                default=",".join(output_formats),
                show_default=True
            )
            output_formats = [fmt.strip() for fmt in output_formats_str.split(",")]
            model_name = typer.prompt(
                "Enter the LLM model name",
                default=model_name,
                show_default=True
            )
            batch_size = typer.prompt(
                "Enter the batch size (for future use)",
                type=int,
                default=batch_size,
                show_default=True
            )

        # Create the MCQRequest object
        request = MCQRequest(
            topic=topic,
            difficulty=difficulty,
            num_questions=num_questions,
            num_options=num_options,
            target_directory=target_directory,
            output_formats=output_formats,
            model_name=model_name,
            batch_size=batch_size
        )

        # Run the workflow
        asyncio.run(generate_mcqs(request))

    except Exception as e:
        logger.error(f"Failed to generate MCQs: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()