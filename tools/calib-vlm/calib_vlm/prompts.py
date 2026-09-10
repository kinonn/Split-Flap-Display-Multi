"""Reader prompt + report_reading tool schema.

The VLM is only a *reader*: it never decides offsets or persists
anything. It reports, for every module position left to right, the
character it sees and the flap condition. The harness reconciles the
answer against the known module count (spaces/prefix/suffix included)
and does all calibration math in Python.
"""

from __future__ import annotations

READER_SYSTEM = """You read a photograph of a modular split-flap display.

The display has a fixed number of modules arranged left to right. Each
module shows exactly ONE character on its flap (a blank module shows a
plain empty flap = a space). You must report every module position.

Rules that matter:
- Report exactly one entry per module, in left-to-right order, index 0
  first. NEVER trim blank/space modules at the start or end: a blank
  module is a real position and must be reported as a space.
- "char" is a single character from the allowed set. Use a space " "
  for a blank/empty flap. Do not invent characters.
- "condition" describes the mechanical state of that module:
  - "clean": the glyph is fully visible and centred, no flap seam
    crossing it.
  - "half": a horizontal seam cuts the glyph, or the module shows part
    of one character above the seam and part of another below it.
  - "double": two seams, or two partial characters, are visible on the
    module at once.
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
                                "enum": ["clean", "half", "double", "blank",
                                         "unreadable"],
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
