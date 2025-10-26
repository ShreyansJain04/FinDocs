#!/usr/bin/env python
"""Verify FinDocs setup and dependencies."""

import sys
from pathlib import Path


def check_imports():
    """Check if all required packages are importable."""
    print("Checking imports...")
    
    required = [
        ("pydantic", "Pydantic"),
        ("yaml", "PyYAML"),
        ("pandas", "Pandas"),
        ("pyarrow", "PyArrow"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence-Transformers"),
        ("faiss", "FAISS"),
        ("rank_bm25", "rank-bm25"),
        ("unstructured", "unstructured"),
        ("fitz", "PyMuPDF"),
        ("pdfplumber", "pdfplumber"),
        ("docx", "python-docx"),
        ("pptx", "python-pptx"),
        ("nltk", "NLTK"),
        ("click", "Click"),
    ]
    
    missing = []
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All imports successful!")
    return True


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ GPU count: {torch.cuda.device_count()}")
        else:
            print("  ⚠ CUDA not available (will use CPU - slower)")
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")


def check_nltk():
    """Check NLTK data."""
    print("\nChecking NLTK data...")
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            print("  ✓ NLTK punkt tokenizer found")
        except LookupError:
            print("  ⚠ NLTK punkt not found, downloading...")
            nltk.download('punkt', quiet=True)
            print("  ✓ NLTK punkt downloaded")
    except Exception as e:
        print(f"  ✗ Error with NLTK: {e}")


def check_structure():
    """Check project structure."""
    print("\nChecking project structure...")
    
    required_dirs = [
        "config",
        "src/fdocs",
        "tests",
        "docs",
    ]
    
    required_files = [
        "config/default.yaml",
        "src/fdocs/__init__.py",
        "src/fdocs/cli.py",
        "src/fdocs/config.py",
        "src/fdocs/chunk.py",
        "src/fdocs/sentiment.py",
        "src/fdocs/embed.py",
        "src/fdocs/index.py",
        "src/fdocs/sparse.py",
        "src/fdocs/rerank.py",
        "src/fdocs/retrieval.py",
        "requirements.txt",
        "README.md",
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - MISSING")
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")


def test_config_loading():
    """Test config loading."""
    print("\nTesting config loading...")
    try:
        from src.fdocs.config import load_config
        config = load_config()
        print(f"  ✓ Config loaded successfully")
        print(f"    - Chunk size: {config.chunking.target_tokens} tokens")
        print(f"    - Sentiment model: {config.sentiment.model}")
        print(f"    - Embedding model: {config.embeddings.model}")
    except Exception as e:
        print(f"  ✗ Error loading config: {e}")


def main():
    """Run all checks."""
    print("=" * 60)
    print("FinDocs Setup Verification")
    print("=" * 60)
    
    checks = [
        check_imports(),
    ]
    
    check_cuda()
    check_nltk()
    check_structure()
    test_config_loading()
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ Setup verification complete! Ready to ingest documents.")
        print("\nNext steps:")
        print("  1. Place documents in docs/<company>/docs*/")
        print("  2. Run: python -m src.fdocs.cli ingest --company <name>")
        print("  3. Query: python -m src.fdocs.cli query --company <name> --query '...'")
        print("\nSee QUICKSTART.md for detailed instructions.")
    else:
        print("❌ Setup incomplete. Please install missing dependencies.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()

