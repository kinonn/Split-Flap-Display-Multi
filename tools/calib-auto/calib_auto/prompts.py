"""Reader prompt + report_reading tool schema.

The VLM is only a *reader*: it never decides offsets or persists
anything. It reports, for every module position left to right, the
character it sees and the flap condition. The harness reconciles the
answer against the known module count (spaces/prefix/suffix included)
and does all calibration math in Python.
"""

from __future__ import annotations

READER_SYSTEM_TEMPLATE = """You read a photograph of a modular split-flap display.

The display has exactly {total} modules arranged left to right. Each
module shows exactly ONE character on its flap (a blank module shows a
plain empty flap = a space). You must report all {total} module positions.
{charset_block}
Rules that matter:
- Report exactly one entry per module, in left-to-right order, index 0
  first. NEVER trim blank/space modules at the start or end: a blank
  module is a real position and must be reported as a space. Your answer
  must contain exactly {total} entries, no more, no fewer.
- "char" must be one of the allowed characters listed above. Use a space
  " " for a blank/empty flap. Do not invent characters.
- "condition" describes whether the module can be read:
  - "clean": the glyph is readable.
  - "blank": the module shows an empty/plain flap.
  - "unreadable": you cannot tell what is shown (blur, glare, angle).
- "confidence" is 0..1 for your reading of MY CHARACTER (not the
  condition).
- Never guess to be helpful: when unsure use condition "unreadable"
  and a low confidence.
- If the photo does not show a display at all, mark every module
  unreadable.
- Always answer by calling the report_reading tool; no prose answer is
  accepted.
"""

# Lookalike guidance shared by the prompt builder and its tests: both
# sides reference this constant, so the prompt text and the assertion can
# never drift apart (the literal in the old tests silently missed the ':'
# the builder had added for the CHARSET_48 drum).
LOOKALIKE_HINT = ("Lookalikes: I/1, O/0, S/5, Z/2/:, B/8, G/6, ./'/- — "
                  "check carefully before choosing.")


def _charset_block(charset: str, drum: str) -> str:
    """Human-readable character-set hint for the system prompt.

    Groups the allowed characters so the model can tell confusable
    glyphs apart (see LOOKALIKE_HINT) and knows the small punctuation
    marks (', ., -, /, :, !, ?, $, @, #, %) are real characters, not dirt
    or seams.
    """
    if not charset:
        return ""
    shown = charset.replace(" ", "\u2423")
    letters = "".join(c for c in charset if c.isalpha())
    digits = "".join(c for c in charset if c.isdigit())
    punct = "".join(c for c in charset if not c.isalnum() and c != " ")
    lines = [f"Allowed characters ({len(charset)}): {shown!r} "
             f"(\u2423 = space/blank)."]
    if letters:
        lines.append(f"Letters: {' '.join(letters)}")
    if digits:
        lines.append(f"Digits: {' '.join(digits)}")
    if punct:
        lines.append(f"Punctuation: {' '.join(punct)} "
                     f"(tiny marks — do not mistake for seams or dirt)")
    lines.append(LOOKALIKE_HINT)
    if drum and drum != charset:
        lines.append(f"Drum order: {drum!r}")
    return "\n" + "\n".join(lines) + "\n"


def reader_system_text(total: int, charset: str = "",
                       drum: str = "") -> str:
    """System prompt with the exact module count filled in.

    The reader already knows `total` on every call, so the system
    message states it explicitly instead of "a fixed number" — the
    count is repeated in the user message too, but models weight the
    system prompt more strongly against blank-trimming.
    """
    return READER_SYSTEM_TEMPLATE.format(
        total=total, charset_block=_charset_block(charset, drum))


# Backwards-compatible generic text (no specific count); prefer
# reader_system_text(total) for actual reads.
READER_SYSTEM = READER_SYSTEM_TEMPLATE.format(
    total="a fixed number of", charset_block="")

REPORT_READING_TOOL = {
    "type": "function",
    "function": {
        "name": "report_reading",
        "description": "Report the character and flap condition of every "
                       "display module, left to right, blanks included.",
        "parameters": {
            "type": "object",
            "properties": {
                "modules": {
                    "type": "array",
                    "description": "One entry per module, index 0 first, "
                                   "exactly as many entries as modules.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "module": {"type": "integer",
                                       "description": "0-based position from the left"},
                            "char": {"type": "string",
                                     "description": "single character, space for blank"},
                            "condition": {
                                "type": "string",
                                "enum": ["clean", "blank", "unreadable"],
                            },
                            "confidence": {"type": "number"},
                        },
                        "required": ["module", "char", "condition"],
                    },
                },
            },
            "required": ["modules"],
        },
    },
}


def reader_user_text(total: int, charset: str, drum: str) -> str:
    allowed = charset.replace(" ", "\u2423") if " " in charset else charset
    return (
        f"The display has exactly {total} modules. Read all {total} module "
        f"positions left to right and call report_reading.\n"
        f"Allowed characters: {allowed!r} (\u2423 = space/blank)\n"
        f"Drum order (characters the flaps can show, in order): {drum!r}\n"
        "Remember: report blanks as space entries, never trim them."
    )


def correction_text(total: int, got: int) -> str:
    return (
        f"Your last answer had {got} module entries but the display has "
        f"exactly {total} modules. Every module position must be present, "
        f"including blank ones. Call report_reading again with exactly "
        f"{total} entries in left-to-right order."
    )
