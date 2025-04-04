# Course Generator Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Architecture](#architecture)
   - [Workflow Engine](#1-workflow-engine-quantalogic-flow)
   - [Processing Steps](#2-main-workflow-steps)
4. [API Reference](#api-reference)
5. [Troubleshooting](#troubleshooting)
6. [Contributing](#contributing)

## Overview
This document explains the architecture and workflow of the AI Course Generator that uses QuantaLogic Flow for content generation.

```mermaid
graph TD
    %% Define styles first
    classDef startNode fill:#4CAF50,stroke:#388E3C,color:white
    classDef processNode fill:#2196F3,stroke:#1976D2,color:white
    classDef decisionNode fill:#FFC107,stroke:#FFA000
    classDef endNode fill:#F44336,stroke:#D32F2F,color:white

    %% Define nodes with simplified names
    S([Start]):::startNode
    V[Validate Input]:::processNode
    T[Generate Title]:::processNode
    O[Generate Outline]:::processNode
    L[Chapter Loop]:::processNode
    M{More Chapters?}:::decisionNode
    C[Compile Course]:::processNode
    F[Format Conversion]:::processNode
    E([End]):::endNode

    %% Define connections
    S --> V
    V --> T
    T --> O
    O --> L
    L --> M
    M -->|Yes| L
    M -->|No| C
    C --> F
    F --> E
```

## Getting Started
### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from ai_course_generator import generate_course

result = generate_course(
    subject="Python Programming",
    chapters=5,
    level="intermediate"
)
```

## Architecture

### 1. Workflow Engine (QuantaLogic Flow)
Key Features:
- Node-based processing pipeline
- Automatic retry mechanism
- Real-time progress tracking

**Example Node Definition**:
```python
@Nodes.define(output="chapter_content")
async def generate_chapter(title, outline, chapter_num):
    """Generates chapter content with error handling"""
    try:
        return await generate_content(title, outline, chapter_num)
    except Exception as e:
        logger.error(f"Chapter generation failed: {e}")
        raise
```

### 2. Main Workflow Steps
```mermaid
sequenceDiagram
    box User
    participant User #FFD54F
    end
    box System
    participant System #42A5F5
    participant QuantaLogic #7E57C2
    end
    box AI Services
    participant LLM #66BB6A
    end
    
    User->>System: Submit Course Request
    System->>QuantaLogic: Create Workflow
    QuantaLogic->>LLM: Generate Title
    LLM-->>QuantaLogic: Title
    QuantaLogic->>LLM: Generate Outline
    LLM-->>QuantaLogic: Outline
    loop For Each Chapter
        QuantaLogic->>LLM: Generate Chapter
        LLM-->>QuantaLogic: Chapter Content
    end
    QuantaLogic->>System: Compiled Course
    System->>User: Final Output
```

## API Reference
### generate_course()
```python
def generate_course(
    subject: str,
    chapters: int,
    level: str = "intermediate",
    output_formats: List[str] = ["pdf", "docx"]
) -> Dict[str, Path]:
    """
    Generates complete course materials
    
    Args:
        subject: Course subject/topic
        chapters: Number of chapters to generate
        level: Difficulty level (beginner|intermediate|advanced)
        output_formats: List of output formats
    
    Returns:
        Dictionary mapping formats to file paths
    """
```

## Troubleshooting
### Common Issues
1. **Pandoc Not Found**
   ```bash
   sudo apt-get install pandoc
   ```

2. **LaTeX Missing**
   ```bash
   sudo apt-get install texlive-latex-base
   ```

3. **Chapter Generation Fails**
   - Check API key permissions
   - Verify network connectivity
   - Retry with reduced chapter length

## Contributing
1. Fork the repository
2. Create feature branch
3. Submit pull request

## Dependencies
```mermaid
graph LR
    classDef main fill:#8D6E63,stroke:#6D4C41,color:white
    classDef service fill:#26C6DA,stroke:#00ACC1
    classDef dependency fill:#FFA726,stroke:#FB8C00
    
    A[Course Generator]:::main --> B[QuantaLogic]:::service
    A --> C[LiteLLM]:::service
    A --> D[Pandoc]:::dependency
    A --> E[LaTeX]:::dependency
    B --> F[AsyncIO]:::dependency
    C --> G[LLM Providers]:::service
```

**Version**: 1.0.0  
**Last Updated**: 2025-04-04
