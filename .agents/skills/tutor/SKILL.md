---
name: tutor
description: Turn a lesson from a local file, URL, or AMAT5315 weekly PDF into an interactive guided tutoring session. Use when the user asks to learn, study, or be taught lesson material.
---

# Tutor

Ground the session in the supplied lesson.

## Load the lesson

- For a local file or direct URL, use that source.
- For “week N,” find the PDF identified as week N under `https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/`. Use the discovered link; do not guess a filename. If no matching PDF is available, report that explicitly.
- For any PDF, run `python3 .agents/skills/tutor/scripts/extract_pdf.py '<path-or-url>'` and read all extracted text before tutoring. The script downloads remote PDFs and extracts them with `pypdf`.
- For other local files, read the file. For other URLs, fetch and read the page.

## Tutor interactively

1. Identify the lesson's goal, prerequisites, and main sections from the source.
2. Start with a brief explanation of the goal, then ask one or two short questions to assess the learner's background.
3. Teach one section at a time using a mix of concise explanation, Socratic questions, and worked examples. Wait for the learner's response before continuing.
4. When an answer is wrong, give a targeted hint and guide the learner through another attempt. Explain the step directly if the learner remains stuck.
5. Keep examples and claims tied to the lesson. Clearly label any extra context introduced to aid understanding.
6. End with a short recap and one check-for-understanding question.
