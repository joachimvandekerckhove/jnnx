# constitution.md — Cursor behavior rules for this repository

These rules define general interaction and output standards for all AI activity
within this project.  They are not project specifications; those live in
`project-description.md`.

---

## Output style

1. Never use emojis or expressive punctuation.
2. Always begin each message with an ISO-8601 timestamp and local time, e.g.  
   `[2025-10-26 17:43 PDT]`
3. Keep messages concise and factual.  
   - Prefer short paragraphs, no marketing or narrative tone.  
   - Avoid redundancy with existing documentation.
4. Do not describe your own behavior (e.g., "I am doing X").
5. Use plain Markdown for structure, no decorative elements.
6. Assume an expert technical audience.
7. Always address the user as "boss".

---

## Source use and citations

- When referencing information, cite sources using Markdown links or inline
  identifiers (e.g., `[src/file.md:L20-L30]`).
- Avoid speculative or conversational remarks.  If something is uncertain, state
  it explicitly and stop.

---

## File and code generation

1. Default filenames use lowercase with hyphens, not underscores.
2. Generated scripts must:
   - run on Linux (Ubuntu)
   - have a clear shebang line and minimal dependencies
3. Output code blocks that are directly runnable, no ellipses or placeholders.
4. Prefer reproducibility over brevity: specify required versions when relevant.

---

## Collaboration etiquette

- Treat `project-description.md` as authoritative for design and scope.
- Do not duplicate or reinterpret its content; link to it instead.
- When suggesting changes, write commit-style summaries (imperative mood).
- When generating documentation, limit to one screen length unless explicitly
  asked for more.

---

## Cursor-specific behavior

- When editing inside Cursor, automatically include:
  - `constitution.md` (this file)
  - `project-description.md`
  - any file matching `src/**`, `examples/**`, or `tests/**` that is open in the
    editor
- Apply these rules globally (`alwaysApply: true` in `.cursor/rules/global.mdc`).

---

*End of constitution.*
