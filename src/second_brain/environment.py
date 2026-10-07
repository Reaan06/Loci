"""Literal configuration parser for environment settings."""

from collections.abc import MutableMapping
import os
from pathlib import Path
import re
import stat

_STRICT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


MAX_ENV_FILE_SIZE = 64 * 1024  # 64 KiB
READ_LIMIT = MAX_ENV_FILE_SIZE + 1


class EnvironmentFileNotFoundError(FileNotFoundError, ValueError):
    """Raised when an explicit environment file is not found."""


def load_environment_file(
    path: Path | str | None = None,
    environment: MutableMapping[str, str] | None = None,
    *,
    required: bool | None = None,
) -> Path | None:
    """Load configuration from a trusted local environment file.

    All standard and custom environment variables defined in the file are loaded
    into the target environment mapping using literal syntax rules. Pre-existing
    keys in the environment (including empty strings) take precedence and are not
    overwritten.

    Security considerations:
        Loading arbitrary environment variables (such as HTTP_PROXY, LD_PRELOAD,
        or PATH) can influence process execution. For this reason, environment files
        must only be sourced from trusted local storage. The loader rejects symlinks,
        directories, FIFOs, files exceeding 64 KiB, and invalid UTF-8 sequences.
        No parent directory traversal is performed.

    Args:
        path: Path to environment file. If None, defaults strictly to ``Path.cwd() / ".env"``.
        environment: Mutable mapping to update (defaults to ``os.environ``).
        required: If True, missing file raises EnvironmentFileNotFoundError.
            If False, missing file is ignored. If None, defaults to False when
            ``path`` is None (default file optional), and True when ``path`` is explicitly provided.

    Returns:
        The Path of the loaded file if loaded, or None if the default file was absent.

    Raises:
        EnvironmentFileNotFoundError: If a required environment file is missing.
        ValueError: If the file is not a regular file, is a symlink, exceeds 64 KiB,
            contains invalid UTF-8 or syntax errors, or cannot be read.
    """
    if path is None:
        target_path = Path.cwd() / ".env"
        is_required = False if required is None else required
    elif isinstance(path, (str, Path, os.PathLike)):
        target_path = Path(path)
        is_required = True if required is None else required
    else:
        err = ValueError("Invalid environment file")
        err.__cause__ = None
        raise err

    flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    fd: int | None = None
    error_to_raise: Exception | None = None
    text: str | None = None

    try:
        try:
            fd = os.open(target_path, flags)
        except FileNotFoundError:
            if os.path.islink(target_path):
                error_to_raise = ValueError("Invalid environment file")
            elif is_required:
                error_to_raise = EnvironmentFileNotFoundError("Environment file not found")
            else:
                return None
        except OSError:
            error_to_raise = ValueError("Invalid environment file")
        except ValueError:
            error_to_raise = ValueError("Invalid environment file")

        if fd is not None:
            try:
                st = os.fstat(fd)
                if not stat.S_ISREG(st.st_mode):
                    error_to_raise = ValueError("Invalid environment file")
                elif st.st_size > MAX_ENV_FILE_SIZE:
                    error_to_raise = ValueError("Invalid environment file")
                else:
                    chunks: list[bytes] = []
                    total_read = 0
                    while total_read < READ_LIMIT:
                        chunk = os.read(fd, READ_LIMIT - total_read)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        total_read += len(chunk)

                    if total_read > MAX_ENV_FILE_SIZE:
                        error_to_raise = ValueError("Invalid environment file")
                    else:
                        raw_data = b"".join(chunks)
                        try:
                            text = raw_data.decode("utf-8")
                        except UnicodeDecodeError:
                            error_to_raise = ValueError("Invalid environment file")
            except OSError:
                error_to_raise = ValueError("Invalid environment file")
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    if error_to_raise is not None:
        error_to_raise.__cause__ = None
        raise error_to_raise

    if text is None:
        return None

    parsed = parse_environment(text)

    if environment is None:
        environment = os.environ

    for key, value in parsed.items():
        if key not in environment:
            environment[key] = value

    return target_path




def parse_environment(text: str) -> dict[str, str]:
    """Parse literal environment configuration text into a key-value dictionary.

    Args:
        text: Environment configuration text.

    Returns:
        Mapping of ASCII keys to literal string values.

    Raises:
        ValueError: If configuration contains duplicate keys, NUL bytes, or invalid syntax.
    """
    if not isinstance(text, str):
        raise ValueError("Invalid environment configuration: input must be text")

    result: dict[str, str] = {}

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        if "\0" in raw_line:
            raise ValueError(f"Invalid environment configuration at line {lineno}")

        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if "=" not in stripped:
            raise ValueError(f"Invalid environment configuration at line {lineno}")

        key_raw, val_raw = stripped.split("=", 1)
        key = key_raw.strip()

        if not _STRICT_KEY_RE.match(key):
            raise ValueError(f"Invalid environment configuration at line {lineno}")

        if key in result:
            raise ValueError(f"Invalid environment configuration at line {lineno}")

        val = val_raw.strip()

        if val.startswith(('"', "'")):
            quote_char = val[0]
            closing_idx = val.find(quote_char, 1)
            if closing_idx == -1:
                raise ValueError(f"Invalid environment configuration at line {lineno}")
            remainder = val[closing_idx + 1:]
            if remainder.strip():
                raise ValueError(f"Invalid environment configuration at line {lineno}")
            parsed_val = val[1:closing_idx]
        elif '"' in val or "'" in val:
            raise ValueError(f"Invalid environment configuration at line {lineno}")
        else:
            parsed_val = val

        result[key] = parsed_val

    return result
