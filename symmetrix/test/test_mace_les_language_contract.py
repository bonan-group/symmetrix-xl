import re
import subprocess
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LANGUAGE_TEST_PATH = "symmetrix/test/test_mace_les_language_contract.py"
UPSTREAM_MODULE_PATH = "mace.modules.extensions"
RETIRED_IDENTIFIERS = (
    "LESCorrection",
    "les_correction",
    "les_extension",
    "symmetrix.extension.les",
    "les_enabled",
    "enable_les",
    "evaluate_les_extension",
)


@dataclass(frozen=True)
class LanguageViolation:
    line: int
    rule: str
    excerpt: str

    def render(self, relative: str) -> str:
        return f"{relative}:{self.line}: {self.rule}: {self.excerpt}"


def _tracked_first_party_text():
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    for relative_bytes in completed.stdout.split(b"\0"):
        if not relative_bytes:
            continue
        relative = relative_bytes.decode()
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            continue
        contents = path.read_bytes()
        if b"\0" in contents:
            continue
        try:
            yield relative, contents.decode()
        except UnicodeDecodeError:
            continue


def _line_number(contents: str, offset: int) -> int:
    return contents.count("\n", 0, offset) + 1


def _logical_text(contents: str) -> tuple[str, list[int]]:
    """Collapse wrapped whitespace while retaining source-line locations."""

    output = []
    source_lines = []
    line = 1
    index = 0
    while index < len(contents):
        if not contents[index].isspace():
            output.append(contents[index])
            source_lines.append(line)
            index += 1
            continue

        start_line = line
        start = index
        while index < len(contents) and contents[index].isspace():
            if contents[index] == "\n":
                line += 1
            index += 1
        whitespace = contents[start:index]
        paragraph_break = re.search(r"\n[ \t\r]*\n", whitespace) is not None
        separator = (
            ". " if paragraph_break and output and output[-1] not in ".!?;" else " "
        )
        for character in separator:
            output.append(character)
            source_lines.append(start_line)
    return "".join(output), source_lines


_STRING_LITERAL = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_LITERAL_JOIN = re.compile(r"(?:\s+|\s*\+\s*|\)\s*\+\s*)")


def _literal_value(token: str) -> str | None:
    body = token[1:-1]
    if "\\" in body:
        return None
    return body


def _split_identifier_violations(contents: str) -> list[LanguageViolation]:
    literals = list(_STRING_LITERAL.finditer(contents))
    violations = []
    for first_index, first in enumerate(literals):
        combined = _literal_value(first.group(0))
        if combined is None:
            continue
        previous = first
        for following in literals[first_index + 1 :]:
            gap = contents[previous.end() : following.start()]
            if _LITERAL_JOIN.fullmatch(gap) is None:
                break
            value = _literal_value(following.group(0))
            if value is None:
                break
            combined += value
            previous = following
            for identifier in RETIRED_IDENTIFIERS:
                if identifier.casefold() in combined.casefold():
                    violations.append(
                        LanguageViolation(
                            line=_line_number(contents, first.start()),
                            rule=f"split-retired-identifier:{identifier}",
                            excerpt=contents[first.start() : following.end()].strip(),
                        )
                    )
    return violations


def _language_violations(relative: str, contents: str) -> list[LanguageViolation]:
    masked = contents.replace(UPSTREAM_MODULE_PATH, "_" * len(UPSTREAM_MODULE_PATH))
    violations = []
    for identifier in RETIRED_IDENTIFIERS:
        for match in re.finditer(re.escape(identifier), masked, re.IGNORECASE):
            violations.append(
                LanguageViolation(
                    line=_line_number(masked, match.start()),
                    rule=f"retired-identifier:{identifier}",
                    excerpt=match.group(0),
                )
            )
    violations.extend(_split_identifier_violations(masked))

    logical, source_lines = _logical_text(masked)
    relation = (
        r"(?:activation|activate(?:d|s|ing)?|correction|extension|"
        r"enable(?:d|s|ing)?|disable(?:d|s|ing)?|optional)"
    )
    les_term = r"(?<![-_\w])LES(?![-_\w])"
    independent_term = r"(?<![-_\w])independent(?![-_\w])"
    model_term = r"(?<![-_\w])model(?![-_\w])"
    sentence_window = r"[^.!?;=:{}\[\]]{0,240}"
    relationship_patterns = (
        re.compile(
            rf"{les_term}{sentence_window}\b{relation}\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b{relation}\b{sentence_window}{les_term}",
            re.IGNORECASE,
        ),
    )
    independent_model_patterns = tuple(
        re.compile(
            sentence_window.join(tokens),
            re.IGNORECASE,
        )
        for tokens in permutations((les_term, independent_term, model_term))
    )
    for rule, patterns in (
        ("misleading-les-relationship", relationship_patterns),
        ("independent-les-model", independent_model_patterns),
    ):
        for pattern in patterns:
            for match in pattern.finditer(logical):
                violations.append(
                    LanguageViolation(
                        line=source_lines[match.start()],
                        rule=rule,
                        excerpt=match.group(0).strip(),
                    )
                )

    return sorted(
        set(violations), key=lambda item: (item.line, item.rule, item.excerpt)
    )


def test_first_party_language_describes_les_as_part_of_maceles():
    violations: list[str] = []
    for relative, contents in _tracked_first_party_text():
        violations.extend(
            violation.render(relative)
            for violation in _language_violations(relative, contents)
            if (relative, violation.line, violation.rule)
            not in LANGUAGE_SCAN_EXEMPTIONS
        )

    assert violations == [], "retired or misleading LES language:\n" + "\n".join(
        violations
    )


@pytest.mark.parametrize(
    ("relative", "text"),
    [
        (
            "mutation.md",
            "LES is an optional correction.",
        ),
        (
            "mutation.py",
            '"""LES is an\noptional correction."""',
        ),
        (
            "mutation.cpp",
            "// LES is presented as a separately configured runtime component with "
            "its own public configuration object and is therefore a correction.",
        ),
        ("mutation.md", "LES is an independent model."),
    ],
)
def test_misleading_les_relationships_are_rejected_across_logical_text(relative, text):
    assert _language_violations(relative, text)


@pytest.mark.parametrize(
    "text",
    [
        "LES remains an independent fitted model.",
        "LES is a model that remains independent.",
        "An independent LES fitted model is described.",
        "An independent fitted model is called LES.",
        "The model describes LES as independent.",
        "The model remains independent and is called LES.",
    ],
)
def test_all_independent_les_model_token_permutations_are_rejected(text):
    assert _language_violations("mutation.md", text)


@pytest.mark.parametrize(
    "text",
    [
        "LES results are documented. The model is independent.",
        "les_model_is_independent",
        "independent-LES-model",
    ],
)
def test_independent_model_rule_respects_boundaries_and_identifiers(text):
    assert _language_violations("admitted.md", text) == []


@pytest.mark.parametrize(
    ("relative", "text"),
    [
        ("mutation.py", 'key = "les_" "extension"'),
        ("mutation.py", 'key = "les_" + "extension"'),
        ("mutation.cpp", 'const char* key = "les_" "extension";'),
        ("mutation.cpp", 'auto key = std::string("les_") + "extension";'),
    ],
)
def test_split_retired_identifiers_are_rejected(relative, text):
    assert _language_violations(relative, text)


@pytest.mark.parametrize(
    "text",
    [
        "from mace.modules.extensions import MACELES",
        "The compiled native extension module loads before evaluation.",
        "MACELESKokkos<Precision> evaluates the complete fitted MACELES model.",
        "LES energy and LES readouts use particle mesh and active reciprocal vectors.",
        "D2 and D3 are independent additive corrections. MACELES is one fitted model.",
    ],
)
def test_accurate_language_and_narrow_exemptions_remain_admitted(text):
    assert _language_violations("admitted.md", text) == []


# Exact exemptions for the contract vocabulary and mutation fixtures in this
# file. Each key is path, physical line, and rule, so unrelated language on the
# same line or anywhere else in this file remains visible to the repository scan.
LANGUAGE_SCAN_EXEMPTIONS = {
    (
        LANGUAGE_TEST_PATH,
        14,
        f"retired-identifier:{RETIRED_IDENTIFIERS[0]}",
    ): "defines the retired-symbol contract",
    (
        LANGUAGE_TEST_PATH,
        15,
        f"retired-identifier:{RETIRED_IDENTIFIERS[1]}",
    ): "defines the retired-symbol contract",
    (
        LANGUAGE_TEST_PATH,
        16,
        f"retired-identifier:{RETIRED_IDENTIFIERS[2]}",
    ): "defines the retired-symbol contract",
    (
        LANGUAGE_TEST_PATH,
        17,
        f"retired-identifier:{RETIRED_IDENTIFIERS[3]}",
    ): "defines the retired-symbol contract",
    (
        LANGUAGE_TEST_PATH,
        18,
        f"retired-identifier:{RETIRED_IDENTIFIERS[4]}",
    ): "defines the retired-symbol contract",
    (
        LANGUAGE_TEST_PATH,
        19,
        f"retired-identifier:{RETIRED_IDENTIFIERS[5]}",
    ): "defines the retired-symbol contract",
    (
        LANGUAGE_TEST_PATH,
        20,
        f"retired-identifier:{RETIRED_IDENTIFIERS[6]}",
    ): "defines the retired-symbol contract",
    (
        LANGUAGE_TEST_PATH,
        20,
        f"retired-identifier:{RETIRED_IDENTIFIERS[2]}",
    ): "the longer retired symbol contains this shorter one",
    (
        LANGUAGE_TEST_PATH,
        211,
        "misleading-les-relationship",
    ): "same-line relationship mutation",
    (
        LANGUAGE_TEST_PATH,
        215,
        "misleading-les-relationship",
    ): "wrapped relationship mutation",
    (
        LANGUAGE_TEST_PATH,
        219,
        "misleading-les-relationship",
    ): "long-window relationship mutation",
    (
        LANGUAGE_TEST_PATH,
        222,
        "independent-les-model",
    ): "independent-model mutation",
    (
        LANGUAGE_TEST_PATH,
        232,
        "independent-les-model",
    ): "LES-independent-model permutation mutation",
    (
        LANGUAGE_TEST_PATH,
        233,
        "independent-les-model",
    ): "LES-model-independent permutation mutation",
    (
        LANGUAGE_TEST_PATH,
        234,
        "independent-les-model",
    ): "independent-LES-model permutation mutation",
    (
        LANGUAGE_TEST_PATH,
        235,
        "independent-les-model",
    ): "independent-model-LES permutation mutation",
    (
        LANGUAGE_TEST_PATH,
        236,
        "independent-les-model",
    ): "model-LES-independent permutation mutation",
    (
        LANGUAGE_TEST_PATH,
        237,
        "independent-les-model",
    ): "model-independent-LES permutation mutation",
}
