<!-- Generated: 2025-06-19 00:28:38 UTC -->

# Architecture

## Overview

The AI Course Generator implements a workflow-driven architecture based on QuantaLogic Flow, organizing content generation through interconnected processing nodes. The system follows a sequential pipeline from input validation through AI-powered content generation to multi-format output production, with automatic retry mechanisms and real-time progress tracking throughout the process.

The architecture separates concerns between workflow orchestration (QuantaLogic), AI integration (LiteLLM), and document processing (Pypandoc), enabling flexible model selection and output format management while maintaining consistent content quality and error handling across all operations.

## Component Map

**Core Workflow Engine:**
- `ai_course_generator/course_generator_agent.py:create_workflow()` - Main workflow construction (line 460)
- `ai_course_generator/course_generator_agent.py:CourseRequest` - Input validation model (lines 72-82)
- `ai_course_generator/course_generator_agent.py:course_stream_observer()` - Progress tracking function

**AI Content Generation Nodes:**
- `generate_title` - LLM node for course title generation (lines 104-108)
- `generate_outline` - Course structure generation (lines 115-125)  
- `generate_chapter` - Individual chapter content creation (lines 170-181)
- `chapter_loop` - Iteration control for multiple chapters (lines 144-169)

**Document Processing Pipeline:**
- `save_full_course` - Content aggregation and file operations (lines 320-356)
- `generate_pdf` - PDF conversion via Pypandoc (lines 213-241)
- `generate_docx` - DOCX format generation (lines 269-291)
- `generate_epub` - EPUB format creation (lines 244-266)

## Key Files

**Workflow Definitions:**
- `ai_course_generator/course_generator_agent.py:Workflow("validate_input")` - Main workflow initialization
- `ai_course_generator/mcq_generator_agent.py:create_mcq_workflow()` - MCQ generation workflow (lines 597-659)

**Node Implementations:**
- `@Nodes.define` decorators - Custom processing nodes throughout course_generator_agent.py
- `@Nodes.llm_node` decorators - AI-powered content generation nodes (lines 94-108, 115-125)

**AI Integration:**
- `course_generator_agent.py:load_system_prompt()` - Dynamic prompt loading (lines 359-377)
- `prompts/*.j2` files - Jinja2 templates for AI model instructions

## Data Flow

**Input Processing:**
1. **CLI Entry** - `generate_course.py:generate()` collects parameters
2. **Validation** - `CourseRequest` model validates inputs (course_generator_agent.py:52-63)
3. **Workflow Creation** - `create_workflow()` builds processing pipeline

**Content Generation Flow:**
1. **System Prompt** - `load_system_prompt()` prepares AI context from prompts/course_system_prompt.j2
2. **Title Generation** - LLM node processes course_title_generation_prompt.j2
3. **Outline Creation** - Structured course outline from course_outline_prompt.j2  
4. **Chapter Loop** - Iterative content generation using course_chapter_prompt.j2
5. **Content Compilation** - `save_full_course()` aggregates all generated content

**Output Processing:**
1. **Format Selection** - Based on CourseRequest boolean flags (pdf_generation, docx_generation, epub_generation)
2. **Document Conversion** - Pypandoc processes Markdown to target formats
3. **File Management** - `clean_title_for_filename()` ensures safe file naming (lines 297-319)