import re


def on_page_markdown(markdown, **kwargs):
    """Strip 'notest' from code fence info strings before rendering."""
    return re.sub(r"^(```\w+) notest$", r"\1", markdown, flags=re.MULTILINE)
