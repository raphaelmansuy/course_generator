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
from datetime import datetime
from typing import List, Optional

import typer
from loguru import logger
from pydantic import BaseModel, Field, validator
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
    num_questions: int = Field(..., description="Number of questions to generate", ge=1, le=1000)
    num_options: int = Field(4, description="Number of options per question", ge=2, le=10)
    target_directory: str = Field(..., description="Directory to save the output files")
    output_formats: List[str] = Field(default=["json"], description="Formats to save the MCQs: json, md")
    model_name: str = Field("gemini/gemini-2.0-flash", description="LLM model to use")
    batch_size: int = Field(5, description="Number of questions to generate in a single batch", ge=1, le=50)
    question_type_distribution: dict = Field(
        default={"memorization": 0.3, "comprehension": 0.4, "deep_understanding": 0.3},
        description="Distribution of question types (proportions must sum to 1): memorization, comprehension, deep_understanding"
    )
    correct_answer_mode_distribution: dict = Field(
        default={"single": 1.0, "multiple": 0.0},
        description="Distribution of single vs multiple correct answer questions (proportions must sum to 1)"
    )
    institution: Optional[str] = Field(None, description="Optional institution style")
    subject_area: str = Field("general", description="Subject area, e.g., computer_science, humanities, sciences")
    include_code_examples: bool = Field(False, description="Force inclusion of code examples in questions")

    @validator("difficulty")
    def validate_difficulty(cls, v):
        if v not in ["beginner", "intermediate", "advanced"]:
            raise ValueError("Difficulty must be beginner, intermediate, or advanced")
        return v

    @validator("output_formats")
    def validate_output_formats(cls, v):
        supported = {"json", "md"}
        if not all(fmt in supported for fmt in v):
            raise ValueError(f"Output formats must be in {supported}")
        return v

    @validator("question_type_distribution", "correct_answer_mode_distribution")
    def validate_proportions(cls, v):
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError("Proportions must sum to 1.0 (tolerance: 0.01)")
        return v

    @validator("target_directory")
    def validate_directory(cls, v):
        dir_path = os.path.dirname(v) or "."
        if not os.path.isdir(dir_path) or not os.access(dir_path, os.W_OK):
            raise ValueError("Target directory must be valid and writable")
        return v

class KeyConcepts(BaseModel):
    """Model for key concepts extracted from a topic."""
    concepts: List[str] = Field(..., min_items=5, description="At least 5 key concepts related to the topic")

class QuestionWithAnswer(BaseModel):
    """Intermediate model for a question and its correct answers."""
    question: str
    correct_answers: List[str]  # List of correct answer texts

class Distractors(BaseModel):
    """Intermediate model for distractor options."""
    distractors: List[str]

class MCQItem(BaseModel):
    """Model for a complete MCQ item."""
    question: str
    options: List[str]
    correct_answers: List[int]  # List of 1-based indices
    explanation: str = ""  # Populated with detailed explanation
    key_concept: str = Field("", description="The key concept this question relates to")
    question_type: str = Field("", description="Type of question: memorization, comprehension, deep_understanding")
    question_type_mode: str = Field("single", description="Mode of correct answers: single or multiple")

class Explanation(BaseModel):
    """Model for the explanation of an MCQ item."""
    explanation: str

# === Helper Functions ===
def assign_question_types(num_questions: int, distribution: dict) -> List[str]:
    """Assign question types or answer modes based on the given distribution."""
    types = list(distribution.keys())
    probs = list(distribution.values())
    if num_questions < len(types):  # Ensure at least one of each type if possible
        assignments = []
        for i in range(num_questions):
            assignments.append(types[i % len(types)])
        random.shuffle(assignments)
    else:
        assignments = random.choices(types, weights=probs, k=num_questions)
    return assignments

# === Workflow Nodes ===
@Nodes.define(output="request")
async def validate_input(request: MCQRequest) -> MCQRequest:
    """Validate the MCQRequest parameters."""
    logger.info(f"Validating input: {request}")
    return request  # Pydantic validation handles all checks

@Nodes.structured_llm_node(
    system_prompt="You are an AI assistant tasked with identifying key concepts for educational content.",
    output="key_concepts",
    response_model=KeyConcepts,
    prompt_template="""
Given the topic '{{ topic }}', identify 5-10 distinct key concepts that are essential for understanding it at the {{ difficulty }} level. Ensure the concepts cover different aspects of the topic and are not overlapping. For advanced difficulty, focus on complex, university-level concepts requiring deep understanding. Provide only the list of concepts, no explanations.
""",
    max_tokens=500
)
async def extract_key_concepts(topic: str, difficulty: str, model: str) -> KeyConcepts:
    """Extract diverse key concepts from the topic using an LLM with retries."""
    for attempt in range(3):
        try:
            return await extract_key_concepts.__wrapped__(topic=topic, difficulty=difficulty, model=model)
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to extract concepts: {e}")
            if attempt == 2:
                raise ValueError("Failed to extract key concepts after 3 attempts")
            await asyncio.sleep(1)

@Nodes.define(output=None)
async def initialize_mcq_generation(request: MCQRequest, key_concepts: KeyConcepts, context: dict) -> None:
    """Initialize the context for MCQ generation with key concepts, question types, and answer modes."""
    logger.info("Initializing MCQ generation")
    num_questions = request.num_questions
    concepts = key_concepts.concepts
    questions_per_concept = [num_questions // len(concepts)] * len(concepts)
    for i in range(num_questions % len(concepts)):
        questions_per_concept[i] += 1
    concept_per_question = []
    for j, concept in enumerate(concepts):
        concept_per_question.extend([concept] * questions_per_concept[j])
    question_types = assign_question_types(num_questions, request.question_type_distribution)
    answer_modes = assign_question_types(num_questions, request.correct_answer_mode_distribution)
    context.update({
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
        "question_types": question_types,
        "answer_modes": answer_modes,
        "progress_task": None,
        "batch_size": request.batch_size,
        "concept_per_question": concept_per_question,
        "question_type_distribution": request.question_type_distribution,
        "correct_answer_mode_distribution": request.correct_answer_mode_distribution,
        "institution": request.institution,
        "subject_area": request.subject_area,
        "include_code_examples": request.include_code_examples,
        "semaphore": asyncio.Semaphore(10)  # Limit concurrency
    })

@Nodes.structured_llm_node(
    system_prompt="You are an AI assistant tasked with generating educational multiple-choice questions.",
    output="question_with_answer",
    response_model=QuestionWithAnswer,
    prompt_template="""
Generate a multiple-choice question of type '{{ question_type }}' on the specific concept '{{ current_concept }}' within the topic '{{ topic }}' at the {{ difficulty }} level. Focus on a unique aspect or subtopic within '{{ current_concept }}' to ensure diversity across questions. The question should have {{ question_type_mode }} correct answer(s). Provide the question and a list of {{ num_correct_answers }} correct answers.

- For 'memorization', create a question that tests recall of facts, definitions, or specific details.
- For 'comprehension', create a question that tests understanding by requiring the student to interpret, explain, or identify the main ideas related to the concept, rather than just recalling facts.
- For 'deep_understanding', create a question that requires higher-order thinking, such as applying the concept to a new situation, analyzing its implications, evaluating different approaches, or synthesizing information from multiple sources. The question should be challenging and reflect the rigor expected at institutions like Cambridge, Oxford, or UCL. Where appropriate, incorporate real-world scenarios, case studies, or data interpretation (e.g., graphs, tables) to add complexity.

{% if subject_area == "computer_science" or include_code_examples %}
Include a full code example relevant to the question. The code should be complete, functional, and demonstrate the concept being tested. Wrap it in triple backticks and specify the programming language, e.g., ```python for Python code.
{% endif %}

{% if institution %}
Incorporate elements in the style of {{ institution }}: 
- For technical subjects, focus on precise reasoning and practical applications
- For computer science specifically:
  * Include code snippets where relevant (properly formatted in markdown)
  * Ask about algorithm complexity or optimization
  * Ask about interpretation of a code program
  * Test debugging skills with incorrect code examples
  * Cover multiple programming paradigms
- For humanities, emphasize critical analysis and nuanced perspectives
- For sciences, include research-oriented elements and data interpretation
{% endif %}

- If 'multiple', ensure the correct answers are distinct but related to the concept.
- For questions with multiple correct answers, ensure the question stem clearly indicates to 'select all that apply' or similar.
- Ensure the question is clear, unambiguous, and provides all necessary information without extraneous details. Use precise, academic language suitable for university students.
""",
    max_tokens=1500  # Increased to accommodate code examples
)
async def generate_question(
    topic: str,
    difficulty: str,
    model: str,
    current_concept: str,
    question_type: str,
    question_type_mode: str,
    num_correct_answers: int,
    institution: Optional[str],
    subject_area: str,
    include_code_examples: bool
) -> QuestionWithAnswer:
    """Generate a question and its correct answers with retries, including code examples when appropriate."""
    for attempt in range(3):
        try:
            return await generate_question.__wrapped__(
                topic=topic, difficulty=difficulty, model=model, current_concept=current_concept,
                question_type=question_type, question_type_mode=question_type_mode,
                num_correct_answers=num_correct_answers, institution=institution,
                subject_area=subject_area, include_code_examples=include_code_examples
            )
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to generate question: {e}")
            if attempt == 2:
                raise ValueError("Failed to generate question after 3 attempts")
            await asyncio.sleep(1)

@Nodes.structured_llm_node(
    system_prompt="Generate plausible distractors for a multiple-choice question.",
    output="distractors",
    response_model=Distractors,
    prompt_template="""
For the question: '{{ question }}' with correct answers: {{ correct_answers }}, generate {{ num_distractors }} plausible distractors that reflect common misunderstandings or errors related to the concept '{{ current_concept }}' within the topic '{{ topic }}'. Ensure they are believable but incorrect, and distinct from all correct answers. Also, ensure the distractors are concise and mutually exclusive, avoiding overlap or confusion with the correct answers.
""",
    max_tokens=1000
)
async def generate_distractors(
    question: str,
    correct_answers: List[str],
    num_distractors: int,
    topic: str,
    current_concept: str,
    model: str
) -> Distractors:
    """Generate distractors for the question with retries."""
    for attempt in range(3):
        try:
            return await generate_distractors.__wrapped__(
                question=question, correct_answers=correct_answers, num_distractors=num_distractors,
                topic=topic, current_concept=current_concept, model=model
            )
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to generate distractors: {e}")
            if attempt == 2:
                raise ValueError("Failed to generate distractors after 3 attempts")
            await asyncio.sleep(1)

@Nodes.define(output="mcq_item")
async def create_mcq_item(
    question_with_answer: QuestionWithAnswer,
    distractors: Distractors,
    current_concept: str,
    question_type: str,
    question_type_mode: str
) -> MCQItem:
    """Create an MCQ item by combining the question, correct answers, distractors, key concept, and question type."""
    logger.debug(f"Creating MCQ item for question: {question_with_answer.question}")
    options = question_with_answer.correct_answers + distractors.distractors
    if len(set(options)) != len(options):
        raise ValueError("Options contain duplicates")
    random.shuffle(options)
    
    correct_indices = []
    for ans in question_with_answer.correct_answers:
        idx = options.index(ans) + 1  # 1-based index
        correct_indices.append(idx)
    
    mcq_item = MCQItem(
        question=question_with_answer.question,
        options=options,
        correct_answers=correct_indices,
        key_concept=current_concept,
        question_type=question_type,
        question_type_mode=question_type_mode
    )
    logger.debug(f"Created MCQ item: question={mcq_item.question}, options={mcq_item.options}, correct_answers={mcq_item.correct_answers}")
    return mcq_item

@Nodes.define(output="mcq_item")
async def validate_mcq_item(mcq_item: MCQItem, num_options: int) -> MCQItem:
    """Validate the MCQ item for uniqueness, correctness, and option count."""
    logger.debug(f"Validating MCQ item: {mcq_item.question}")
    if len(set(mcq_item.options)) != len(mcq_item.options):
        raise ValueError(f"Duplicate options in MCQ item: {mcq_item.question}")
    if not mcq_item.options:
        raise ValueError(f"No options provided for MCQ item: {mcq_item.question}")
    if len(mcq_item.options) != num_options:
        raise ValueError(f"Expected {num_options} options, got {len(mcq_item.options)}")
    max_index = len(mcq_item.options)
    for idx in mcq_item.correct_answers:
        if idx < 1 or idx > max_index:
            raise ValueError(f"Invalid correct answer index {idx} (max {max_index}) for: {mcq_item.question}")
    logger.debug(f"Validated MCQ item: question={mcq_item.question}, options={mcq_item.options}, correct_answers={mcq_item.correct_answers}")
    return mcq_item

@Nodes.define(output=None)
async def prepare_explanation_inputs(mcq_item: MCQItem) -> dict:
    """Prepare inputs for generating the explanation."""
    logger.debug(f"Preparing explanation for: question={mcq_item.question}, options={mcq_item.options}, correct_answers={mcq_item.correct_answers}")
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    if not mcq_item.options or not mcq_item.correct_answers:
        raise ValueError(f"Invalid MCQ item: options={mcq_item.options}, correct_answers={mcq_item.correct_answers}")
    
    correct_indices = [idx - 1 for idx in mcq_item.correct_answers]  # Convert to 0-based indices
    valid_indices = [i for i in correct_indices if 0 <= i < len(mcq_item.options)]
    if len(valid_indices) != len(correct_indices):
        raise ValueError(f"Correct indices {correct_indices} out of range for options {mcq_item.options}")
    
    correct_letters = ", ".join([letters[idx] for idx in valid_indices])
    correct_options = ", ".join([mcq_item.options[idx] for idx in valid_indices])
    formatted_options = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(mcq_item.options)])
    logger.debug(f"Prepared: correct_letters={correct_letters}, correct_options={correct_options}")
    return {
        "formatted_options": formatted_options,
        "correct_letters": correct_letters,
        "correct_options": correct_options
    }

@Nodes.structured_llm_node(
    system_prompt="You are an AI assistant tasked with providing detailed explanations for multiple-choice questions.",
    output="explanation",
    response_model=Explanation,
    prompt_template="""
Here is a multiple-choice question: '{{ question }}'

Options:
{{ formatted_options }}

The correct answers are: {{ correct_letters }} ({{ correct_options }}).

{% if "```" in question %}
Note: The question includes a code example. Ensure your explanation references the code and explains how it relates to the correct answer(s).
{% endif %}

Provide a detailed explanation (at least 100 words) suitable for university-level students, explaining why the correct answers are right and why each distractor is incorrect, referencing specific concepts or common misconceptions. Use the actual option letters (e.g., A, B, C) and their text in your explanation. Do not include placeholders like '{correct_letters}' or '{explanation}' in your response—use the provided values directly.
""",
    max_tokens=1500  # Increased to accommodate code-related explanations
)
async def generate_explanation(
    question: str,
    formatted_options: str,
    correct_letters: str,
    correct_options: str,
    model: str
) -> Explanation:
    """Generate an explanation for the MCQ with retries and length check, referencing code if present."""
    for attempt in range(3):
        try:
            explanation = await generate_explanation.__wrapped__(
                question=question, formatted_options=formatted_options,
                correct_letters=correct_letters, correct_options=correct_options, model=model
            )
            if len(explanation.explanation.split()) < 100:
                logger.warning(f"Explanation too short ({len(explanation.explanation.split())} words)")
                if attempt == 2:
                    raise ValueError("Explanation too short after 3 attempts")
                continue
            return explanation
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to generate explanation: {e}")
            if attempt == 2:
                raise ValueError("Failed to generate explanation after 3 attempts")
            await asyncio.sleep(1)

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

@Nodes.define(output="saved_mcqs")
async def save_mcqs(
    mcq_items: List[MCQItem],
    output_formats: List[str],
    target_directory: str,
    topic: str,
    difficulty: str,
    num_questions: int,
    num_options: int,
    model_name: str,
    question_type_distribution: dict,
    correct_answer_mode_distribution: dict,
    institution: Optional[str],
    subject_area: str,
    include_code_examples: bool
) -> None:
    """Save the MCQ items to files in the specified formats with metadata and redundancy checks."""
    logger.info(f"Saving {len(mcq_items)} MCQs to {target_directory}")
    
    # Check for redundancy
    question_texts = [item.question for item in mcq_items]
    if len(question_texts) != len(set(question_texts)):
        logger.warning("Duplicate question texts found. Consider reviewing for redundancy.")
    correct_answers_sets = [frozenset(item.correct_answers) for item in mcq_items]
    if len(correct_answers_sets) != len(set(correct_answers_sets)):
        logger.warning("Questions with identical correct answers found. Consider reviewing for redundancy.")
    
    os.makedirs(target_directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metadata = {
        "topic": topic,
        "difficulty": difficulty,
        "num_questions": num_questions,
        "num_options": num_options,
        "model_name": model_name,
        "question_type_distribution": question_type_distribution,
        "correct_answer_mode_distribution": correct_answer_mode_distribution,
        "institution": institution,
        "subject_area": subject_area,
        "include_code_examples": include_code_examples,
        "generated_at": timestamp
    }

    for fmt in output_formats:
        file_path = os.path.join(target_directory, f"mcqs_{timestamp}.{fmt}")
        if os.path.exists(file_path):
            if not typer.confirm(f"File {file_path} exists. Overwrite?"):
                file_path = os.path.join(target_directory, f"mcqs_{timestamp}_new.{fmt}")
        if fmt == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "metadata": metadata,
                    "mcqs": [item.dict() for item in mcq_items]
                }, f, indent=2)
        elif fmt == "md":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# MCQ Generation Metadata\n\n")
                f.write(f"- **Topic**: {topic}\n")
                f.write(f"- **Difficulty**: {difficulty}\n")
                f.write(f"- **Number of Questions**: {num_questions}\n")
                f.write(f"- **Number of Options per Question**: {num_options}\n")
                f.write(f"- **Model Name**: {model_name}\n")
                if institution:
                    f.write(f"- **Institution Style**: {institution.capitalize()}\n")
                f.write(f"- **Subject Area**: {subject_area.capitalize()}\n")
                f.write(f"- **Include Code Examples**: {include_code_examples}\n")
                f.write("- **Question Type Distribution**:\n")
                for qtype, prop in question_type_distribution.items():
                    f.write(f"  - {qtype.capitalize()}: {prop * 100:.0f}%\n")
                f.write("- **Correct Answer Mode Distribution**:\n")
                for mode, prop in correct_answer_mode_distribution.items():
                    f.write(f"  - {mode.capitalize()}: {prop * 100:.0f}%\n")
                f.write(f"- **Generated At**: {timestamp}\n")
                f.write("\n---\n\n")
                f.write("# Multiple Choice Questions\n\n")
                for idx, item in enumerate(mcq_items, 1):
                    f.write(f"{idx}. **Question**: {item.question}\n\n")
                    for opt_idx, opt in enumerate(item.options):
                        f.write(f"   - {chr(65 + opt_idx)}. {opt}\n")
                    correct_letters = ", ".join([chr(65 + i - 1) for i in item.correct_answers])
                    f.write(f"\n   **Correct Answer(s)**: {correct_letters}\n")
                    f.write(f"   **Explanation**: {item.explanation}\n")
                    f.write(f"   **Key Concept**: {item.key_concept}\n")
                    f.write(f"   **Question Type**: {item.question_type}\n")
                    f.write(f"   **Answer Mode**: {item.question_type_mode}\n\n")

@Nodes.define(output=None)
async def generate_mcq_batch(context: dict) -> None:
    """Generate a batch of MCQ items concurrently with error handling."""
    batch_size = context["batch_size"]
    num_questions = context["num_questions"]
    current_index = context["current_index"]
    end_index = min(current_index + batch_size, num_questions)
    question_indices = range(current_index, end_index)
    
    logger.info(f"Generating batch: questions {current_index} to {end_index - 1}")
    
    async def bounded_task(task):
        async with context["semaphore"]:
            return await task
    
    tasks = [
        bounded_task(asyncio.create_task(
            generate_single_question(
                topic=context["topic"],
                difficulty=context["difficulty"],
                model=context["model"],
                current_concept=context["concept_per_question"][i],
                question_type=context["question_types"][i],
                question_type_mode=context["answer_modes"][i],
                num_correct_answers=1 if context["answer_modes"][i] == "single" else 2,
                institution=context["institution"],
                subject_area=context["subject_area"],
                include_code_examples=context["include_code_examples"],
                num_options=context["num_options"]
            )
        )) for i in question_indices
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    mcq_items_batch = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Question {current_index + i} failed: {result}")
        else:
            mcq_items_batch.append(result)
    
    context["mcq_items"].extend(mcq_items_batch)
    context["current_index"] = end_index
    
    progress = context.get("progress")
    task = context.get("progress_task")
    if progress and task:
        progress.update(task, completed=end_index)

# === Helper Function for Batch Generation ===
async def generate_single_question(
    topic: str,
    difficulty: str,
    model: str,
    current_concept: str,
    question_type: str,
    question_type_mode: str,
    num_correct_answers: int,
    institution: Optional[str],
    subject_area: str,
    include_code_examples: bool,
    num_options: int
) -> MCQItem:
    """Generate a single MCQ item with question, distractors, and explanation, including code examples when specified."""
    question_with_answer = await generate_question(
        topic=topic, difficulty=difficulty, model=model, current_concept=current_concept,
        question_type=question_type, question_type_mode=question_type_mode,
        num_correct_answers=num_correct_answers, institution=institution,
        subject_area=subject_area, include_code_examples=include_code_examples
    )
    if len(question_with_answer.correct_answers) != num_correct_answers:
        raise ValueError(f"Expected {num_correct_answers} correct answers, got {len(question_with_answer.correct_answers)}")
    
    distractors = await generate_distractors(
        question=question_with_answer.question, correct_answers=question_with_answer.correct_answers,
        num_distractors=num_options - num_correct_answers, topic=topic, current_concept=current_concept, model=model
    )
    if len(set(distractors.distractors)) != len(distractors.distractors):
        raise ValueError("Distractors contain duplicates")
    if any(d in question_with_answer.correct_answers for d in distractors.distractors):
        raise ValueError("Distractors overlap with correct answers")
    
    mcq_item = await create_mcq_item(
        question_with_answer=question_with_answer, distractors=distractors,
        current_concept=current_concept, question_type=question_type, question_type_mode=question_type_mode
    )
    mcq_item = await validate_mcq_item(mcq_item=mcq_item, num_options=num_options)
    explanation_inputs = await prepare_explanation_inputs(mcq_item=mcq_item)
    explanation = await generate_explanation(
        question=mcq_item.question, formatted_options=explanation_inputs["formatted_options"],
        correct_letters=explanation_inputs["correct_letters"], correct_options=explanation_inputs["correct_options"],
        model=model
    )
    mcq_item = await set_explanation(mcq_item=mcq_item, explanation=explanation)
    return mcq_item

# === Workflow Construction ===
def create_mcq_workflow() -> Workflow:
    """Construct the workflow for generating MCQs with batch processing."""
    wf = Workflow("validate_input")
    
    wf.node("validate_input").then("extract_key_concepts")
    wf.node("extract_key_concepts").then("initialize_mcq_generation")
    wf.node("initialize_mcq_generation").then("generate_mcq_batch")
    wf.node("generate_mcq_batch").then("save_mcqs")
    
    wf.transitions["generate_mcq_batch"] = [
        ("generate_mcq_batch", lambda ctx: ctx["current_index"] < ctx["num_questions"]),
        ("save_mcqs", lambda ctx: ctx["current_index"] >= ctx["num_questions"])
    ]
    
    wf.node_input_mappings["validate_input"] = {"request": "request"}
    wf.node_input_mappings["extract_key_concepts"] = {
        "topic": lambda ctx: ctx["request"].topic,
        "difficulty": lambda ctx: ctx["request"].difficulty,
        "model": lambda ctx: ctx["request"].model_name
    }
    wf.node_input_mappings["initialize_mcq_generation"] = {
        "request": "request",
        "key_concepts": "key_concepts",
        "context": lambda ctx: ctx
    }
    wf.node_input_mappings["generate_mcq_batch"] = {"context": lambda ctx: ctx}
    wf.node_input_mappings["save_mcqs"] = {
        "mcq_items": "mcq_items",
        "output_formats": "output_formats",
        "target_directory": "target_directory",
        "topic": "topic",
        "difficulty": "difficulty",
        "num_questions": "num_questions",
        "num_options": "num_options",
        "model_name": "model",
        "question_type_distribution": "question_type_distribution",
        "correct_answer_mode_distribution": "correct_answer_mode_distribution",
        "institution": "institution",
        "subject_area": "subject_area",
        "include_code_examples": "include_code_examples"
    }
    
    logger.info("Batch MCQ workflow created")
    return wf

# === Observer for Progress Tracking ===
async def mcq_stream_observer(event: WorkflowEvent):
    """Observer to track and display workflow progress with a progress bar and colored output."""
    total = event.context.get("num_questions", 0)
    index = event.context.get("current_index", 0)
    
    if event.node_name == "initialize_mcq_generation":
        with Progress() as progress:
            task = progress.add_task("[yellow]Generating MCQs", total=total)
            event.context["progress"] = progress
            event.context["progress_task"] = task
    
    elif event.node_name == "generate_mcq_batch":
        progress = event.context.get("progress")
        task = event.context.get("progress_task")
        if progress and task:
            progress.update(task, completed=index, description="[yellow]Generating batch")
    
    elif event.node_name == "save_mcqs":
        console.print(f"[green]Saving MCQs to {event.context['target_directory']}")

# === Workflow Execution ===
async def generate_mcqs(request: MCQRequest):
    """Run the MCQ generation workflow with comprehensive error handling."""
    logger.info(f"Starting MCQ generation for topic: {request.topic}")
    workflow = create_mcq_workflow()
    engine = workflow.build()
    engine.add_observer(mcq_stream_observer)
    initial_context = {"request": request}
    try:
        result = await engine.run(initial_context)
        logger.info("MCQ generation completed")
        console.print(f"[green]MCQs successfully generated and saved to {request.target_directory}")
        return result
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        console.print(f"[red]Generation failed: {str(e)}[/red]")
        raise

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
    output_formats: List[str] = typer.Option(["json"], help="Output formats (e.g., json, md)"),
    model_name: str = typer.Option("gemini/gemini-2.0-flash", help="LLM model name"),
    batch_size: int = typer.Option(5, help="Number of questions to generate per batch"),
    memorization: float = typer.Option(0.3, help="Proportion of memorization questions (0.0 to 1.0)"),
    comprehension: float = typer.Option(0.4, help="Proportion of comprehension questions (0.0 to 1.0)"),
    deep_understanding: float = typer.Option(0.3, help="Proportion of deep understanding questions (0.0 to 1.0)"),
    single: float = typer.Option(0.5, help="Proportion of single correct answer questions (0.0 to 1.0)"),
    multiple: float = typer.Option(0.5, help="Proportion of multiple correct answer questions (0.0 to 1.0)"),
    institution: Optional[str] = typer.Option(None, help="Institution style: any string"),
    subject_area: str = typer.Option("general", help="Subject area, e.g., computer_science, humanities, sciences"),
    include_code_examples: bool = typer.Option(False, help="Force inclusion of code examples in questions")
):
    """Generate MCQs and save them to the specified directory. Enters interactive mode if parameters are omitted."""
    try:
        interactive = topic is None or difficulty is None or num_questions is None

        if interactive:
            console.print("[bold cyan]Entering interactive mode...[/bold cyan]")
            topic = typer.prompt("Enter the topic for the MCQs")
            difficulty = typer.prompt(
                "Enter the difficulty level (beginner, intermediate, advanced)",
                type=str, default="beginner", show_default=True
            ).lower()
            while difficulty not in ["beginner", "intermediate", "advanced"]:
                console.print("[red]Invalid difficulty level. Please choose: beginner, intermediate, advanced[/red]")
                difficulty = typer.prompt("Enter the difficulty level").lower()
            num_questions = typer.prompt(
                "Enter the number of questions to generate (1-1000)",
                type=int, default=5, show_default=True
            )
            while not (1 <= num_questions <= 1000):
                console.print("[red]Number of questions must be between 1 and 1000[/red]")
                num_questions = typer.prompt("Enter the number of questions", type=int)
            num_options = typer.prompt(
                "Enter the number of options per question (2-10)",
                type=int, default=num_options, show_default=True
            )
            while not (2 <= num_options <= 10):
                console.print("[red]Number of options must be between 2 and 10[/red]")
                num_options = typer.prompt("Enter the number of options", type=int)
            target_directory = typer.prompt(
                "Enter the target directory for output files",
                default=target_directory, show_default=True
            )
            output_formats_str = typer.prompt(
                "Enter output formats (comma-separated, e.g., json, md)",
                default=",".join(output_formats), show_default=True
            )
            output_formats = [fmt.strip() for fmt in output_formats_str.split(",")]
            model_name = typer.prompt(
                "Enter the LLM model name",
                default=model_name, show_default=True
            )
            batch_size = typer.prompt(
                "Enter the batch size for generation (1-50)",
                type=int, default=batch_size, show_default=True
            )
            while not (1 <= batch_size <= 50):
                console.print("[red]Batch size must be between 1 and 50[/red]")
                batch_size = typer.prompt("Enter the batch size", type=int)
            memorization = typer.prompt(
                "Enter the proportion of memorization questions (0.0 to 1.0)",
                type=float, default=memorization, show_default=True
            )
            comprehension = typer.prompt(
                "Enter the proportion of comprehension questions (0.0 to 1.0)",
                type=float, default=comprehension, show_default=True
            )
            deep_understanding = typer.prompt(
                "Enter the proportion of deep understanding questions (0.0 to 1.0)",
                type=float, default=deep_understanding, show_default=True
            )
            single = typer.prompt(
                "Enter the proportion of single correct answer questions (0.0 to 1.0)",
                type=float, default=single, show_default=True
            )
            multiple = typer.prompt(
                "Enter the proportion of multiple correct answer questions (0.0 to 1.0)",
                type=float, default=multiple, show_default=True
            )
            institution = typer.prompt(
                "Enter the institution style (any string)",
                default="", show_default=False
            ).strip() or None
            subject_area = typer.prompt(
                "Enter the subject area (e.g., computer_science, humanities, sciences)",
                default=subject_area, show_default=True
            )
            include_code_examples = typer.confirm(
                "Include code examples in questions?",
                default=include_code_examples
            )

        total_qtype_proportion = memorization + comprehension + deep_understanding
        if abs(total_qtype_proportion - 1.0) > 0.01:
            console.print("[red]Error: Question type proportions must sum to 1.0[/red]")
            raise ValueError("Question type proportions must sum to 1.0")

        total_mode_proportion = single + multiple
        if abs(total_mode_proportion - 1.0) > 0.01:
            console.print("[red]Error: Correct answer mode proportions must sum to 1.0[/red]")
            raise ValueError("Correct answer mode proportions must sum to 1.0")

        request = MCQRequest(
            topic=topic,
            difficulty=difficulty,
            num_questions=num_questions,
            num_options=num_options,
            target_directory=target_directory,
            output_formats=output_formats,
            model_name=model_name,
            batch_size=batch_size,
            question_type_distribution={
                "memorization": memorization,
                "comprehension": comprehension,
                "deep_understanding": deep_understanding
            },
            correct_answer_mode_distribution={
                "single": single,
                "multiple": multiple
            },
            institution=institution,
            subject_area=subject_area,
            include_code_examples=include_code_examples
        )

        asyncio.run(generate_mcqs(request))

    except Exception as e:
        logger.error(f"Failed to generate MCQs: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()