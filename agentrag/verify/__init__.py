"""Scientific claim verification on SciFact: retrieve, judge, critique, retry.

Everything that talks to the Gemini API is isolated behind ``gemini.py`` and
cached on disk (``cache.py``), so the unit tests and CI never need a key or
network and reruns of a finished experiment are free. Heavy dependencies
(google-genai, transformers) are imported lazily; install them with
``pip install -e ".[verify]"``.
"""
