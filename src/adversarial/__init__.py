"""
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
