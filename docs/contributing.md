# Contributing Guide

## 🤝 How to Contribute

### Getting Started

1. **Fork the Repository**
   - Click the 'Fork' button on GitHub
   - Clone your fork locally

2. **Set Up Development Environment**
   ```bash
   git clone https://github.com/your-username/data-analysis-titanic.git
   cd data-analysis-titanic
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Process

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use type hints
   - Write descriptive docstrings
   - Keep functions focused and small

2. **Testing**
   - Write unit tests for new features
   - Ensure all tests pass
   - Maintain or improve code coverage
   ```bash
   pytest tests/
   pytest --cov=src tests/
   ```

3. **Documentation**
   - Update relevant documentation
   - Add docstrings to new functions
   - Include code examples where helpful

### Submitting Changes

1. **Commit Messages**
   ```
   feat: Add new visualization for age distribution
   ^--^  ^------------------------------------^
   |     |
   |     +-> Summary in present tense
   |
   +-------> Type: feat, fix, docs, style, refactor, test, chore
   ```

2. **Pull Request Process**
   - Update your fork
   - Create a pull request
   - Fill out the PR template
   - Request review from maintainers

3. **Code Review**
   - Address review comments
   - Update PR as needed
   - Be responsive to feedback

## 🎯 Project Structure

```
src/
├── application/     # Business logic
├── domain/         # Core entities
├── infrastructure/ # External interfaces
└── utils/          # Helper functions
```

## 🔍 Code Review Guidelines

### What We Look For

1. **Code Quality**
   - Clean and readable code
   - Proper error handling
   - Efficient implementations
   - No duplicate code

2. **Testing**
   - Comprehensive test coverage
   - Edge cases handled
   - Clear test descriptions
   - Proper use of fixtures

3. **Documentation**
   - Clear and concise comments
   - Updated documentation
   - Proper docstrings
   - Code examples

## 🐛 Bug Reports

### What to Include

1. **Environment**
   - OS version
   - Python version
   - Package versions
   - Browser (for dashboard issues)

2. **Steps to Reproduce**
   - Clear sequence of steps
   - Example code if applicable
   - Input data if relevant

3. **Expected vs Actual**
   - What you expected
   - What actually happened
   - Screenshots if relevant

## 🚀 Feature Requests

### Proposal Guidelines

1. **Problem Statement**
   - Clear description of the problem
   - Use cases
   - Current limitations

2. **Proposed Solution**
   - Detailed description
   - Technical approach
   - Implementation ideas

3. **Impact Assessment**
   - Benefits
   - Potential drawbacks
   - Performance considerations

## 📊 Project Boards

### Board Structure

1. **To Do**
   - Upcoming features
   - Known issues
   - Documentation needs

2. **In Progress**
   - Currently being worked on
   - Assigned to contributors
   - Under review

3. **Done**
   - Completed features
   - Merged PRs
   - Closed issues

## 🎉 Recognition

- All contributors are listed in CONTRIBUTORS.md
- Significant contributions are highlighted
- Contributors are credited in release notes
