# Contributing to utils_cmip7

This guide covers the development workflow, testing procedures, and contribution guidelines for the `utils_cmip7` project.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

---

## Development Setup

### Prerequisites

- Python ≥ 3.9
- Git
- Virtual environment tool (venv, conda, etc.)

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Climateyousheng/utils_cmip7.git
   cd utils_cmip7
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install in editable mode with dev dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify installation:**
   ```bash
   python -c "import utils_cmip7; print(utils_cmip7.__version__)"
   pytest tests/ -v
   ```

### Environment Variables

Optional configuration:

```bash
# Custom RECCAP2 mask path
export UTILS_CMIP7_RECCAP_MASK=/path/to/custom/mask.nc

# Default: ~/scripts/hadcm3b-ensemble-validator/observations/RECCAP_AfricaSplit_MASK11_Mask_regridded.hadcm3bl_grid.nc
```

---

## Development Workflow

### Branch Strategy

- `main` - Stable release branch
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Typical Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make changes and test:**
   ```bash
   # Edit code
   pytest tests/ -v
   ```

3. **Run code formatters:**
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

4. **Check code quality:**
   ```bash
   flake8 src/ tests/
   mypy src/
   ```

5. **Commit changes:**
   ```bash
   git add .
   git commit -m "feat: Add new feature description"
   ```

6. **Push and create PR:**
   ```bash
   git push origin feature/my-new-feature
   # Create PR on GitHub
   ```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation only
- `refactor` - Code refactoring
- `test` - Adding tests
- `chore` - Build/tooling changes
- `perf` - Performance improvements
- `ci` - CI/CD changes

**Examples:**
```
feat: Add ensemble parameter auto-detection to CLI
fix: Handle missing NetCDF dimensions in level selection
docs: Update CLI reference with auto-detection examples
test: Add integration tests for ensemble loader
```

---

## Testing

### Running Tests

**Run all tests:**
```bash
pytest tests/ -v
```

**Run specific test file:**
```bash
pytest tests/test_cli_helpers.py -v
```

**Run with coverage:**
```bash
pytest tests/ -v --cov=src/utils_cmip7 --cov-report=html
```

**View coverage report:**
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Test Structure

```
tests/
├── test_imports.py                    # Package import tests
├── test_config.py                     # Configuration tests
├── test_canonical_variables.py        # Variable registry tests
├── test_cli_helpers.py                # CLI helper function tests
├── test_cli_auto_detection.py         # CLI auto-detection tests
├── test_io/                           # I/O layer tests
│   ├── test_extract.py
│   ├── test_stash.py
│   └── test_file_discovery.py
├── test_processing/                   # Processing layer tests
│   ├── test_spatial.py
│   ├── test_temporal.py
│   └── test_metrics.py
├── test_diagnostics/                  # Diagnostics tests
│   └── test_extraction.py
├── test_validation/                   # Validation tests
│   ├── test_compare.py
│   └── test_outputs.py
└── test_plotting/                     # Plotting tests
    └── test_maps.py
```

### Writing Tests

**Unit test example:**
```python
import pytest
from utils_cmip7.cli import _extract_ensemble_prefix

def test_extract_prefix_five_chars():
    """Test extraction from 5-character experiment ID."""
    assert _extract_ensemble_prefix('xqjca') == 'xqjc'

def test_extract_prefix_short():
    """Test that short IDs are returned as-is."""
    assert _extract_ensemble_prefix('abc') == 'abc'
```

**Integration test example:**
```python
import pytest
from pathlib import Path
from utils_cmip7.validation import load_ensemble_params_from_logs

@pytest.fixture
def mock_logs(tmp_path):
    """Create mock ensemble logs for testing."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    # Create mock log file
    return log_dir

def test_load_params(mock_logs):
    """Test loading parameters from logs."""
    params = load_ensemble_params_from_logs(str(mock_logs), 'xqjc')
    assert 'xqjca' in params
```

### Test Coverage Goals

- **Core modules** (io, processing, diagnostics): 80%+ coverage
- **Validation module**: 70%+ coverage
- **CLI**: 50%+ coverage (integration tests focus)
- **Plotting**: 30%+ coverage (visual inspection focus)

Current overall coverage: ~35% (416+ tests)

---

## Code Style

### Python Code Formatting

This project uses:
- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

### Configuration

Settings in `pyproject.toml`:

```toml
[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311', 'py312']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

### Running Formatters

**Format all code:**
```bash
black src/ tests/
isort src/ tests/
```

**Check without modifying:**
```bash
black --check src/ tests/
isort --check-only src/ tests/
```

**Lint code:**
```bash
flake8 src/ tests/
```

**Type check:**
```bash
mypy src/
```

### Pre-commit Setup (Optional)

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

---

## Documentation

### Documentation Structure

```
docs/
├── API.md                             # Complete API reference
├── CLI_REFERENCE.md                   # CLI command documentation
├── CONTRIBUTING.md                    # This file
├── MIGRATION_GUIDE.md                 # Version migration guide
├── TROUBLESHOOTING.md                 # Common issues
├── VALIDATION_EXPLAINED.md            # Validation methodology
├── VALIDATION_METHODS_QUICK_REFERENCE.md
└── STASH.md                           # STASH code reference
```

### Updating Documentation

**When to update docs:**
- Adding new CLI commands → Update `CLI_REFERENCE.md`
- Changing API → Update `API.md`
- Breaking changes → Update `MIGRATION_GUIDE.md` and `CHANGELOG.md`
- New features → Update `CHANGELOG.md` and relevant docs

**Documentation checklist:**
- [ ] Update docstrings in code
- [ ] Update relevant markdown files
- [ ] Add examples if needed
- [ ] Update CHANGELOG.md
- [ ] Update version numbers if needed

### Docstring Style

Use NumPy-style docstrings:

```python
def extract_ensemble_prefix(expt_id: str) -> str:
    """
    Extract ensemble prefix from experiment ID.

    Convention: 5-character IDs have 4-character prefix (xqjca → xqjc)

    Parameters
    ----------
    expt_id : str
        Experiment identifier

    Returns
    -------
    str
        Ensemble prefix for log file matching

    Examples
    --------
    >>> extract_ensemble_prefix('xqjca')
    'xqjc'
    >>> extract_ensemble_prefix('xqhuc')
    'xqhu'

    Notes
    -----
    IDs longer or shorter than 5 characters are returned as-is.
    """
    return expt_id[:4] if len(expt_id) == 5 else expt_id
```

---

## Pull Request Process

### Before Submitting

1. **Ensure tests pass:**
   ```bash
   pytest tests/ -v
   ```

2. **Format code:**
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

3. **Update documentation:**
   - Add docstrings
   - Update relevant markdown files
   - Update CHANGELOG.md

4. **Update version if needed:**
   ```bash
   # In pyproject.toml
   version = "0.4.1"
   ```

### PR Checklist

- [ ] All tests pass
- [ ] Code is formatted (black, isort)
- [ ] Docstrings added/updated
- [ ] CHANGELOG.md updated
- [ ] Documentation updated
- [ ] No breaking changes (or documented in MIGRATION_GUIDE.md)
- [ ] Commit messages follow conventional format

### PR Description Template

```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation update

## Changes
- List key changes
- Include motivation/context

## Testing
- Describe tests added
- Show test coverage

## Checklist
- [ ] Tests pass
- [ ] Code formatted
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

### Review Process

1. CI/CD checks must pass (tests across Python 3.9-3.12)
2. At least one approval from maintainer
3. No unresolved comments
4. Documentation is complete

---

## Release Process

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **Major** (x.0.0) - Breaking changes
- **Minor** (0.x.0) - New features (backward compatible)
- **Patch** (0.0.x) - Bug fixes

### Release Checklist

1. **Update version:**
   ```bash
   # Edit pyproject.toml
   version = "0.4.1"
   ```

2. **Update CHANGELOG.md:**
   ```markdown
   ## [0.4.1] - 2026-02-16

   ### Added
   - Feature description

   ### Fixed
   - Bug fix description
   ```

3. **Run full test suite:**
   ```bash
   pytest tests/ -v --cov=src/utils_cmip7
   ```

4. **Build and test package:**
   ```bash
   pip install build
   python -m build
   pip install dist/utils_cmip7-0.4.1-py3-none-any.whl
   ```

5. **Create git tag:**
   ```bash
   git tag -a v0.4.1 -m "Release v0.4.1"
   git push origin v0.4.1
   ```

6. **Create GitHub release:**
   - Go to GitHub releases
   - Create new release from tag
   - Copy CHANGELOG entry
   - Attach wheel file

### Post-Release

- Update documentation
- Announce in relevant channels
- Monitor issue tracker

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/Climateyousheng/utils_cmip7/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Climateyousheng/utils_cmip7/discussions)
- **Email**: ysli.13477426@gmail.com

---

## Code of Conduct

Be respectful, collaborative, and constructive. We welcome contributions from everyone.

---

Last updated: 2026-02-16
