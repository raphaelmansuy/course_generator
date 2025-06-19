<!-- Generated: 2025-06-19 00:28:38 UTC -->

# Testing

## Overview

The project uses a pytest-based testing approach configured in `pyproject.toml` with CI/CD integration through GitHub Actions. While test files are not currently present in the repository, the testing infrastructure is configured for Python 3.10-3.13 compatibility testing and package build verification across multiple environments.

## Test Types

**Unit Testing Framework:**
- **pytest ^8.0.0** - Testing framework specified in pyproject.toml:24
- **Test Discovery** - Standard pytest discovery patterns (test_*.py, *_test.py)
- **Configuration** - No custom pytest configuration currently present

**Integration Testing:**
- **Workflow Testing** - Potential tests for QuantaLogic Flow workflows in course_generator_agent.py
- **AI Model Integration** - Testing LiteLLM integration with mock responses
- **Document Generation** - Validation of PDF/DOCX/EPUB output formats

**CI/CD Testing:**
- **Matrix Testing** - Python versions 3.10, 3.11, 3.12, 3.13 (.github/workflows/python-tests.yml:10)
- **Build Validation** - `poetry build` verification in CI pipeline
- **Cross-platform** - Ubuntu-latest environment testing

## Running Tests

**Local Development:**
```bash
# Install test dependencies
poetry install

# Run tests (when available)
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=ai_course_generator

# Run specific test file
poetry run pytest tests/test_course_generator.py -v
```

**Development Testing:**
```bash
# Test CLI functionality
poetry run python -m ai_course_generator.generate_course --help

# Test course generation
poetry run generate_course --subject "Test" --level beginner --number-of-chapters 2 --words-by-chapter 500 --target-directory ./test_output

# Test MCQ generation
python -c "from ai_course_generator.mcq_generator_agent import MCQRequest; print('MCQ import successful')"
```

**CI/CD Pipeline** (.github/workflows/python-tests.yml):
```yaml
# Automated testing on push/PR
- name: Install dependencies
  run: poetry install --no-root
- name: Build package  
  run: poetry build
# Note: pytest execution currently commented out (line 24-25)
```

## Reference

**Test File Organization:**
- **tests/** - Standard pytest test directory (not yet created)
- **test_course_generator.py** - Main workflow testing
- **test_mcq_generator.py** - MCQ functionality testing
- **test_document_generation.py** - Output format validation

**Testing Utilities:**
- **Mock AI Responses** - Test without actual API calls to LiteLLM
- **Temporary Directories** - Safe test file generation and cleanup
- **Workflow Fixtures** - Reusable QuantaLogic Flow test setups

**Build System Test Targets:**
- `poetry run pytest` - Execute all tests
- `poetry run black --check .` - Code style validation
- `poetry run isort --check-only .` - Import order verification  
- `poetry run ruff check .` - Linting validation

**Test Environment Setup:**
- **Virtual Environment** - Isolated dependency management via Poetry
- **Test Dependencies** - Separate dev dependencies in pyproject.toml:23-28
- **CI/CD Integration** - Automated testing on repository changes