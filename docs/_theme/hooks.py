"""MkDocs hooks: share text and files with the app instead of copying them.

- ``<!-- help: Topic -->`` in a page is replaced by that topic of the in-app
  help (``tensorboard_book.helptext.HELP``), with its references to other
  topics turned into links.
- ``<!-- cli: COMMAND -->`` is replaced by the ``--help`` output of
  ``tensorboard-book COMMAND`` (``<!-- cli: -->`` for the main command).
- The logo and the fonts are published from the package, so the site uses
  the same files as the app.
- The footer shows the copyright years as ``--version`` does.
- Help text written for the app gets the blank line that Python-Markdown
  needs before a list, so its lists render as lists.
"""

from __future__ import annotations

import contextlib
import io
import os
import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File

from tensorboard_book import cli
from tensorboard_book.helptext import HELP

PACKAGE = Path(__file__).parents[2] / "src" / "tensorboard_book"
PUBLISHED = {
    "assets/logo.svg": PACKAGE / "assets" / "logo.svg",
    "assets/logo-dark.svg": PACKAGE / "assets" / "logo-dark.svg",
    **{
        f"assets/fonts/{font.name}": font
        for font in sorted((PACKAGE / "static" / "fonts").iterdir())
    },
}
# Where each help topic is in the docs, for "see the *Filters* topic".
TOPIC_PAGES = {
    "Filters": "topics/filters.md",
    "Status and time": "topics/status-and-time.md",
    "Properties": "topics/properties.md",
    "TensorBoard": "topics/tensorboard.md",
    "Password and access": "deployment.md",
}
HELP_MARK = re.compile(r"<!-- help: (.+?) -->")
CLI_MARK = re.compile(r"<!-- cli:\s*(\S*)\s*-->")
TOPIC_REF = re.compile(r"the \*([^*]+)\* topic")
# A line of text directly followed by a list. Streamlit renders the list,
# but Python-Markdown needs a blank line between them.
LIST_START = re.compile(
    r"^(?![-*] |\d+\. )(\S.*)\n(?=[-*] |\d+\. )", re.MULTILINE
)


def on_config(config):
    """Set the footer's copyright line."""
    config["copyright"] = (
        f"© {cli.copyright_years()} {config['site_author']}."
        " <code>tensorboard-book</code> is released under the MIT License."
    )
    return config


def on_files(files, config):
    """Add the logo and fonts from the package to the site."""
    for uri, path in PUBLISHED.items():
        files.append(File.generated(config, uri, abs_src_path=str(path)))
    return files


def on_page_markdown(markdown, page, config, files):
    """Fill in the help topics and command line help."""
    here = posixpath.dirname(page.file.src_uri)

    def link(match: re.Match) -> str:
        target = TOPIC_PAGES.get(match.group(1))
        if target is None:
            return match.group(0)
        return f"[{match.group(1)}]({posixpath.relpath(target, here)})"

    def help_topic(match: re.Match) -> str:
        text = HELP[match.group(1)].strip()
        text = TOPIC_REF.sub(link, text)
        text = LIST_START.sub(r"\1\n\n", text)
        # The app points to the documentation; link the section.
        target = posixpath.relpath("deployment.md", here)
        return text.replace(
            "See *Password and deployment* in the documentation",
            f"See [Access from another computer]"
            f"({target}#access-from-another-computer)",
        )

    markdown = HELP_MARK.sub(help_topic, markdown)
    return CLI_MARK.sub(lambda m: command_help(m.group(1)), markdown)


def command_help(command: str) -> str:
    """Return ``tensorboard-book COMMAND --help`` as a code block."""
    out = io.StringIO()
    os.environ["COLUMNS"] = "79"
    with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
        cli.main([command, "--help"] if command else ["--help"])
    return f"```text\n{out.getvalue().rstrip()}\n```"
