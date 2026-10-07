"""Password protection from environment variables.

The password is never stored in a file. Instead, an environment variable holds
a salted scrypt hash of it, created with ``tensorboard-book hash-password``:

* ``TBOOK_PASSWORD_HASH``: full access (browse, download, edit).
* ``TBOOK_VIEWER_PASSWORD_HASH``: optional read-only access (browse and
  download, no edits).

The hash format is ``<salt hex>:<hash hex>``.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time

EDITOR_ENV = "TBOOK_PASSWORD_HASH"
VIEWER_ENV = "TBOOK_VIEWER_PASSWORD_HASH"
NO_AUTH_ENV = "TBOOK_NO_AUTH"


def _scrypt(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32
    )


def hash_password(password: str) -> str:
    """Create the value to put in ``TBOOK_PASSWORD_HASH``.

    Args:
        password: The plain password.

    Returns:
        ``"<salt hex>:<hash hex>"``.
    """
    salt = secrets.token_bytes(16)
    return f"{salt.hex()}:{_scrypt(password, salt).hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a stored hash in constant time.

    Args:
        password: The password that was typed.
        stored: The ``"<salt hex>:<hash hex>"`` value.

    Returns:
        True if the password matches. A malformed hash never matches.
    """
    try:
        salt_hex, hash_hex = stored.strip().split(":")
        expected = bytes.fromhex(hash_hex)
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    return hmac.compare_digest(_scrypt(password, salt), expected)


def check_hash_format(stored: str) -> bool:
    """Tell whether a hash value looks well formed.

    Args:
        stored: The environment variable value.

    Returns:
        True if it is two hex strings separated by a colon.
    """
    try:
        salt_hex, hash_hex = stored.strip().split(":")
        return (
            len(bytes.fromhex(salt_hex)) >= 8
            and len(bytes.fromhex(hash_hex)) == 32
        )
    except ValueError:
        return False


def require_login(header=None) -> str:
    """Show a password form in Streamlit until the visitor logs in.

    Nothing after this call runs for visitors who are not logged in, because
    the function calls ``st.stop()``.

    Args:
        header: Optional function that draws the page header (logo and name).

    Returns:
        ``"editor"`` or ``"viewer"``.
    """
    import streamlit as st

    if os.environ.get(NO_AUTH_ENV) == "1":
        return "editor"
    role = st.session_state.get("tbook_role")
    if role:
        return role

    editor_hash = os.environ.get(EDITOR_ENV, "")
    viewer_hash = os.environ.get(VIEWER_ENV, "")
    if not editor_hash:
        st.error(
            f"{EDITOR_ENV} is not set, so the app refuses to start. "
            "Create one with `tensorboard-book hash-password`."
        )
        st.stop()

    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        st.write("")
        if header is not None:
            header(size=56, title_size="2rem")
        else:
            st.title("tensorboard-book")
        with st.form("login"):
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted and password:
            if verify_password(password, editor_hash):
                st.session_state.tbook_role = "editor"
                st.rerun()
            if viewer_hash and verify_password(password, viewer_hash):
                st.session_state.tbook_role = "viewer"
                st.rerun()
            time.sleep(1.0)  # Slows down guessing.
            st.error("Wrong password")
    st.stop()
