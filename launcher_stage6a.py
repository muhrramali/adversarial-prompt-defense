#!/usr/bin/env python3
"""
launcher_stage6a.py
===================
Stage 6 Part A — __init__ — adversarial transforms and test runner.

This launcher will create:
     - src/adversarial/__init__.py

Files are written using r'''...''' raw strings, so content is preserved
byte-for-byte without any encoding/decoding.

Usage:
    python launcher_stage6a.py
"""
from pathlib import Path

FILES = {
    '''src/adversarial/__init__.py''': r'''"""
src/adversarial/__init__.py
"""
from src.adversarial.transforms import (
    TRANSFORMS,
    apply_all_transforms,
    apply_transform,
    paraphrase_synonyms,
    obfuscate_leetspeak,
    case_upper,
    case_lower,
    case_mixed,
    case_manipulate,
    punctuation_insert,
    whitespace_extra,
    whitespace_compact,
    whitespace_manipulate,
    unicode_lookalikes,
    indirect_wrap,
)
from src.adversarial.generator import (
    generate_adversarial_set,
    save_adversarial_set,
    summarize_adversarial_set,
)

__all__ = [
    # transforms
    "TRANSFORMS",
    "apply_all_transforms",
    "apply_transform",
    "paraphrase_synonyms",
    "obfuscate_leetspeak",
    "case_upper",
    "case_lower",
    "case_mixed",
    "case_manipulate",
    "punctuation_insert",
    "whitespace_extra",
    "whitespace_compact",
    "whitespace_manipulate",
    "unicode_lookalikes",
    "indirect_wrap",
    # generator
    "generate_adversarial_set",
    "save_adversarial_set",
    "summarize_adversarial_set",
]
''',
}


def main():
    created, overwritten = 0, 0
    for rel_path, content in FILES.items():
        p = Path(rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        exists = p.exists()
        p.write_text(content, encoding="utf-8")
        if exists:
            overwritten += 1
            print(f"  [UPDATE] {rel_path} ({len(content)} bytes)")
        else:
            created += 1
            print(f"  [NEW]    {rel_path} ({len(content)} bytes)")

    print(f"\nDone! {created} new, {overwritten} updated.")
    print(f"\nNext step: Run:  python launcher_stage6b.py")


if __name__ == "__main__":
    main()