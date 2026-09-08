#!/usr/bin/env python3

import sys
import tempfile
import urllib.request
from pathlib import Path

from pypdf import PdfReader


source = sys.argv[1]

if source.startswith(("http://", "https://")):
    with urllib.request.urlopen(source) as response, tempfile.NamedTemporaryFile(suffix=".pdf") as pdf:
        pdf.write(response.read())
        pdf.flush()
        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(pdf.name).pages)
else:
    text = "\n\n".join(page.extract_text() or "" for page in PdfReader(Path(source)).pages)

if not text.strip():
    raise ValueError("PDF contains no extractable text")

print(text)
