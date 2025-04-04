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
- [Contributing](#contributing)
- [License](#license)

## Key Features

✔ **Multi-format Output** - Generate courses in PDF, DOCX, and EPUB formats  
✔ **Smart Configuration** - Customize length, difficulty, and content depth  
✔ **Model Flexibility** - Supports multiple AI models including Gemini  
✔ **User-Friendly CLI** - Interactive prompts or direct command options  
✔ **Structured Content** - Automatically generates chapters and outlines  

## Quick Start

```bash
# Run in interactive mode (recommended for first-time users)
python -m ai_course_generator.generate_course generate --interactive
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
python -m ai_course_generator.generate_course generate -i
```
The interactive mode will guide you through all configuration options with sensible defaults.

### Direct Command Usage
```bash
python -m ai_course_generator.generate_course generate \
  --subject "Data Science" \
  --level intermediate \
  --words-by-chapter 1200 \
  --no-epub
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
python -m ai_course_generator.generate_course generate \
  --subject "Python Basics" \
  --level beginner \
  --words-by-chapter 600
```

**2. Technical Deep Dive**
```bash
python -m ai_course_generator.generate_course generate \
  --subject "Advanced Kubernetes" \
  --level advanced \
  --words-by-chapter 2000 \
  --model-name "gemini/gemini-2.0-pro"
```

## Contributing

We welcome contributions! Please see our [Contribution Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
