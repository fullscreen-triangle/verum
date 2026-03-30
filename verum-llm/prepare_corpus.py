"""
Prepare training corpus for the Verum-specialized LLM.
Extracts and processes all knowledge from the Verum framework.
"""
import os
import json
import glob
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def extract_latex_papers():
    """Extract text from all LaTeX papers."""
    papers = []
    for tex_path in glob.glob(os.path.join(PROJECT_ROOT, "**/*.tex"), recursive=True):
        with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        # Strip LaTeX commands but keep content
        text = strip_latex(content)
        papers.append({
            "source": os.path.relpath(tex_path, PROJECT_ROOT),
            "type": "paper",
            "text": text,
        })
    return papers


def strip_latex(text):
    """Remove LaTeX formatting, keep mathematical content readable."""
    # Remove comments
    text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
    # Remove \begin{document}...\end{document} wrapper but keep content
    text = re.sub(r'\\documentclass.*?\\begin\{document\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\end\{document\}', '', text)
    # Remove \usepackage, \newcommand etc
    text = re.sub(
        r'\\(usepackage|newcommand|newtheorem|theoremstyle|geometry|hypersetup)\{[^}]*\}(\{[^}]*\})*(\[[^\]]*\])*',
        '', text
    )
    text = re.sub(
        r'\\(usepackage|newcommand|newtheorem)\[[^\]]*\]\{[^}]*\}(\{[^}]*\})*',
        '', text
    )
    # Convert theorem environments to readable format
    text = re.sub(
        r'\\begin\{(theorem|lemma|definition|proposition|corollary|axiom|remark|proof)\}(\[[^\]]*\])?',
        r'\n[\1] ', text
    )
    text = re.sub(
        r'\\end\{(theorem|lemma|definition|proposition|corollary|axiom|remark|proof)\}',
        '\n', text
    )
    # Convert sections
    text = re.sub(r'\\section\{([^}]*)\}', r'\n# \1\n', text)
    text = re.sub(r'\\subsection\{([^}]*)\}', r'\n## \1\n', text)
    text = re.sub(r'\\subsubsection\{([^}]*)\}', r'\n### \1\n', text)
    # Keep equations readable
    text = re.sub(r'\\begin\{equation\}(\[label=[^\]]*\])?', r'\nEQUATION: ', text)
    text = re.sub(r'\\end\{equation\}', '\n', text)
    text = re.sub(r'\\begin\{align\}', r'\nEQUATIONS:\n', text)
    text = re.sub(r'\\end\{align\}', '\n', text)
    # Remove remaining LaTeX commands but keep arguments
    text = re.sub(r'\\textbf\{([^}]*)\}', r'**\1**', text)
    text = re.sub(r'\\textit\{([^}]*)\}', r'*\1*', text)
    text = re.sub(r'\\emph\{([^}]*)\}', r'*\1*', text)
    text = re.sub(r'\\cite\{[^}]*\}', '', text)
    text = re.sub(r'\\ref\{[^}]*\}', '[ref]', text)
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    # Clean up
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_python_code():
    """Extract Python source code with docstrings."""
    code_files = []
    for py_path in glob.glob(os.path.join(PROJECT_ROOT, "**/*.py"), recursive=True):
        if "node_modules" in py_path or ".next" in py_path or "__pycache__" in py_path:
            continue
        with open(py_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        code_files.append({
            "source": os.path.relpath(py_path, PROJECT_ROOT),
            "type": "python",
            "text": content,
        })
    return code_files


def extract_rust_code():
    """Extract Rust source code."""
    rust_files = []
    for rs_path in glob.glob(os.path.join(PROJECT_ROOT, "**/*.rs"), recursive=True):
        with open(rs_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        rust_files.append({
            "source": os.path.relpath(rs_path, PROJECT_ROOT),
            "type": "rust",
            "text": content,
        })
    return rust_files


def extract_markdown():
    """Extract markdown documentation."""
    md_files = []
    for md_path in glob.glob(os.path.join(PROJECT_ROOT, "**/*.md"), recursive=True):
        if "node_modules" in md_path or ".next" in md_path:
            continue
        with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        md_files.append({
            "source": os.path.relpath(md_path, PROJECT_ROOT),
            "type": "markdown",
            "text": content,
        })
    return md_files


def extract_validation_results():
    """Extract all JSON validation results."""
    results = []
    for json_path in glob.glob(os.path.join(PROJECT_ROOT, "**/*.json"), recursive=True):
        if "node_modules" in json_path or ".next" in json_path or "package" in json_path:
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "source": os.path.relpath(json_path, PROJECT_ROOT),
                "type": "validation_result",
                "text": json.dumps(data, indent=2, default=str)[:50000],  # cap size
            })
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return results


def build_corpus():
    """Build the complete training corpus."""
    print("Building Verum training corpus...")

    corpus = []

    print("  Extracting LaTeX papers...")
    corpus.extend(extract_latex_papers())

    print("  Extracting Python code...")
    corpus.extend(extract_python_code())

    print("  Extracting Rust code...")
    corpus.extend(extract_rust_code())

    print("  Extracting Markdown docs...")
    corpus.extend(extract_markdown())

    print("  Extracting validation results...")
    corpus.extend(extract_validation_results())

    # Statistics
    total_chars = sum(len(item["text"]) for item in corpus)
    print(f"\nCorpus statistics:")
    print(f"  Total documents: {len(corpus)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated tokens: ~{total_chars // 4:,}")
    print(f"  By type:")
    for t in sorted(set(item["type"] for item in corpus)):
        count = sum(1 for item in corpus if item["type"] == t)
        chars = sum(len(item["text"]) for item in corpus if item["type"] == t)
        print(f"    {t}: {count} files, {chars:,} chars")

    # Save corpus
    output_path = os.path.join(os.path.dirname(__file__), "corpus.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nCorpus saved to: {output_path}")
    return corpus


if __name__ == "__main__":
    build_corpus()
