# AI Course Generator 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

An intelligent CLI tool for generating structured educational content using AI models.

## Table of Contents
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation Options](#installation-options)
- [Usage Guide](#usage-guide)
  - [Interactive Mode](#interactive-mode)
  - [Direct Command Usage](#direct-command-usage)
- [Configuration Reference](#configuration-reference)
- [Examples](#examples)
- [Technical Documentation](#technical-documentation)
- [Contributing](#contributing)
- [License](#license)

## Key Features

✔ **Multi-format Output** - Generate courses in PDF, DOCX, and EPUB formats  
✔ **Smart Configuration** - Customize length, difficulty, and content depth  
✔ **Model Flexibility** - Supports multiple AI models including Gemini  
✔ **User-Friendly CLI** - Interactive prompts or direct command options  
✔ **Structured Content** - Automatically generates chapters and outlines  
✔ **Advanced Workflows** - Powered by QuantaLogic Flow engine ([Technical Details](TECHNICAL_README.md))

## Quick Start

```bash
# Run in interactive mode (recommended for first-time users)
python -m ai_course_generator.generate_course --interactive
```

## Installation Options

### Standard Installation
```bash
git clone https://github.com/raphaelmansuy/course-generator.git
cd course-generator
pip install -e .
```

### Isolated Installation (recommended)
```bash
pipx install git+https://github.com/raphaelmansuy/course-generator.git
```

## Usage Guide

### Interactive Mode
```bash
python -m ai_course_generator.generate_course --interactive
# or
python -m ai_course_generator.generate_course -i
```
The interactive mode will guide you through all configuration options with sensible defaults.

### Direct Command Usage

#### Required Parameters

All of these must be provided either via command line or interactive mode:
- `--subject`: Course subject (string)
- `--number-of-chapters`: Number of chapters (integer)
- `--level`: Difficulty level (beginner/intermediate/advanced)
- `--words-by-chapter`: Target word count per chapter (integer)
- `--target-directory`: Output directory path (string)

#### Complete Command Example:
```bash
python -m ai_course_generator.generate_course \
  --subject "Python Basics" \
  --number-of-chapters 5 \
  --level beginner \
  --words-by-chapter 800 \
  --target-directory "./output_courses"
```

#### With Optional Flags:
```bash
python -m ai_course_generator.generate_course \
  --subject "Advanced ML" \
  --number-of-chapters 8 \
  --level advanced \
  --words-by-chapter 1500 \
  --target-directory "./ml_courses" \
  --no-pdf \
  --model-name "gemini/gemini-2.0-pro"
```

## Configuration Reference

| Parameter | Description | Values | Default |
|-----------|-------------|--------|---------|
| `--subject` | Course topic | Any string | Required |
| `--level` | Difficulty | beginner/intermediate/advanced | intermediate |
| `--number-of-chapters` | Course length | 1-20 | 5 |
| `--words-by-chapter` | Content depth | 500-5000 | 1000 |
| `--target-directory` | Output path | Valid path | ./courses/[subject] |
| `--model-name` | AI model | Supported model name | gemini/gemini-2.0-flash |

## Examples

**1. Beginner-Friendly Course**
```bash
python -m ai_course_generator.generate_course \
  --subject "Python Basics" \
  --level beginner \
  --words-by-chapter 600
```

**2. Technical Deep Dive**
```bash
python -m ai_course_generator.generate_course \
  --subject "Advanced Kubernetes" \
  --level advanced \
  --words-by-chapter 2000 \
  --model-name "gemini/gemini-2.0-pro"
```

## Technical Documentation
For in-depth technical information about the architecture and implementation, see our [Technical Documentation](TECHNICAL_README.md) which covers:
- Workflow engine design
- AI integration patterns
- Content generation pipeline
- Performance characteristics

## Contributing

We welcome contributions! Please see our [Contribution Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
