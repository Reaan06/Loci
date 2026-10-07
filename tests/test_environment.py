"""Unit tests for literal environment configuration parser."""

import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

from second_brain.environment import (
    EnvironmentFileNotFoundError,
    load_environment_file,
    parse_environment,
)



class TestParseEnvironment(unittest.TestCase):
    """Test suite for parse_environment."""

    def test_empty_and_whitespace_only(self):
        """Empty input and whitespace-only lines return an empty mapping."""
        self.assertEqual(parse_environment(""), {})
        self.assertEqual(parse_environment("   \n\n\t  \n"), {})

    def test_comments_and_blank_lines(self):
        """Full-line comments and blank lines are ignored."""
        text = (
            "# Top comment\n"
            "\n"
            "   # Indented comment\n"
            "FOO=bar\n"
            "# Middle comment\n"
            "BAZ=qux\n"
            "#"
        )
        self.assertEqual(parse_environment(text), {"FOO": "bar", "BAZ": "qux"})

    def test_strict_ascii_keys_and_whitespace_trimming(self):
        """Keys and unquoted values have outer whitespace trimmed."""
        text = "   FOO   =   bar   \n_PRIVATE_KEY=secret_val\nPORT_8080=8080"
        self.assertEqual(
            parse_environment(text),
            {"FOO": "bar", "_PRIVATE_KEY": "secret_val", "PORT_8080": "8080"},
        )

    def test_empty_values(self):
        """Keys with empty or whitespace-only values result in empty string."""
        text = "EMPTY_VAL=\nSPACED_EMPTY=   \nQUOTED_EMPTY=\"\"\nSINGLE_QUOTED_EMPTY=''"
        self.assertEqual(
            parse_environment(text),
            {
                "EMPTY_VAL": "",
                "SPACED_EMPTY": "",
                "QUOTED_EMPTY": "",
                "SINGLE_QUOTED_EMPTY": "",
            },
        )

    def test_quotes_literal_content_and_inner_whitespace(self):
        """Quotes preserve inner whitespace and literal content without escape interpretation."""
        text = (
            'DOUBLE_QUOTED="  spaced value  "\n'
            "SINGLE_QUOTED='  another space  '\n"
            'NO_ESCAPE="line\\nbreak\\tpreserved"\n'
            'NESTED_SINGLE="single\'inside\'double"\n'
            "NESTED_DOUBLE='double\"inside\"single'"
        )
        self.assertEqual(
            parse_environment(text),
            {
                "DOUBLE_QUOTED": "  spaced value  ",
                "SINGLE_QUOTED": "  another space  ",
                "NO_ESCAPE": "line\\nbreak\\tpreserved",
                "NESTED_SINGLE": "single'inside'double",
                "NESTED_DOUBLE": 'double"inside"single',
            },
        )

    def test_unquoted_hash_is_literal_except_full_comment(self):
        """Hash character in unquoted or quoted values is treated literally."""
        text = (
            "HASH_IN_VALUE=foo#bar\n"
            "HASH_WITH_SPACE=foo # bar\n"
            'QUOTED_HASH="foo#bar"\n'
            "QUOTED_HASH_SPACE=\"foo # bar\""
        )
        self.assertEqual(
            parse_environment(text),
            {
                "HASH_IN_VALUE": "foo#bar",
                "HASH_WITH_SPACE": "foo # bar",
                "QUOTED_HASH": "foo#bar",
                "QUOTED_HASH_SPACE": "foo # bar",
            },
        )

    def test_dollar_and_command_literal_no_interpolation(self):
        """Dollar signs, variable references, and commands are preserved literally."""
        text = (
            "VAR_REF=$EXISTING_VAR\n"
            "BRACED_REF=${SOME_VAR:-default}\n"
            "CMD_SUB=$(whoami)\n"
            "BACKTICK_CMD=`id -u`\n"
            "PRICE=$100"
        )
        self.assertEqual(
            parse_environment(text),
            {
                "VAR_REF": "$EXISTING_VAR",
                "BRACED_REF": "${SOME_VAR:-default}",
                "CMD_SUB": "$(whoami)",
                "BACKTICK_CMD": "`id -u`",
                "PRICE": "$100",
            },
        )

    def test_values_with_multiple_equals_signs(self):
        """Values can contain equal signs without being split further."""
        text = "URL=https://example.com/api?user=1&token=xyz\nEQS===double=="
        self.assertEqual(
            parse_environment(text),
            {
                "URL": "https://example.com/api?user=1&token=xyz",
                "EQS": "==double==",
            },
        )

    def test_reject_duplicate_key(self):
        """Duplicate keys must raise ValueError atomically."""
        text = "KEY=first\nOTHER=fine\nKEY=second"
        with self.assertRaises(ValueError):
            parse_environment(text)

    def test_reject_bad_key_syntax(self):
        """Keys not conforming to strict ASCII identifier rules must raise ValueError."""
        bad_inputs = [
            "1LEADING_DIGIT=val",
            "DASH-NOT-ALLOWED=val",
            "SPACE IN KEY=val",
            "DOT.NOT.ALLOWED=val",
            "=MISSING_KEY",
            "NO_EQUALS_SIGN",
            "NON_ASCII_KEY_\u00e9=val",
            "   =VALUE",
        ]
        for bad in bad_inputs:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_environment(bad)

    def test_reject_unmatched_quote(self):
        """Unmatched quotes must raise ValueError."""
        bad_inputs = [
            'UNCLOSED_DOUBLE="value',
            "UNCLOSED_SINGLE='value",
            'LONE_DOUBLE="',
            "LONE_SINGLE='",
            "MISMATCH_DOUBLE_SINGLE=\"value'",
            "MISMATCH_SINGLE_DOUBLE='value\"",
            'STRAY_QUOTE_AT_END=value"',
            "STRAY_SINGLE_AT_END=value'",
            'STRAY_QUOTE_IN_MIDDLE=val"ue',
            "STRAY_SINGLE_IN_MIDDLE=val'ue",
        ]
        for bad in bad_inputs:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_environment(bad)

    def test_reject_trailing_quote_junk(self):
        """Trailing non-whitespace after closing quote must raise ValueError."""
        bad_inputs = [
            'TRAILING_TEXT="val"junk',
            'TRAILING_SPACE_TEXT="val" junk',
            "TRAILING_SINGLE_TEXT='val'junk",
            "TRAILING_SINGLE_SPACE_TEXT='val' junk",
            'DOUBLE_CLOSES_EARLY="hello"world"',
            'TWO_QUOTED_BLOCKS="hello" "world"',
        ]
        for bad in bad_inputs:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_environment(bad)

    def test_reject_export_prefix(self):
        """Bash export syntax is rejected as invalid key / syntax."""
        bad_inputs = [
            "export FOO=bar",
            "export FOO",
            "export   FOO=bar",
        ]
        for bad in bad_inputs:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_environment(bad)

    def test_reject_nul_bytes(self):
        """NUL bytes anywhere in input must raise ValueError."""
        bad_inputs = [
            "FOO=bar\0",
            "FO\0O=bar",
            "\0FOO=bar",
            "# Comment with \0 NUL",
        ]
        for bad in bad_inputs:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_environment(bad)

    def test_errors_do_not_leak_secret_context(self):
        """Error messages must be generic and never leak secret keys, values, or line context."""
        sensitive_key = "TOP_SECRET_AUTH_TOKEN_KEY"
        sensitive_val = "super_confidential_password_987654"
        sensitive_cases = [
            f"{sensitive_key}=\"{sensitive_val}",  # unmatched quote
            f"{sensitive_key}=\"{sensitive_val}\"trailing_secret",  # trailing junk
            f"export {sensitive_key}={sensitive_val}",  # export syntax
            f"{sensitive_key}={sensitive_val}\n{sensitive_key}={sensitive_val}",  # duplicate
            f"{sensitive_key}={sensitive_val}\0",  # NUL byte
            f"1{sensitive_key}={sensitive_val}",  # bad key
        ]
        for case in sensitive_cases:
            with self.subTest(case=case):
                try:
                    parse_environment(case)
                    self.fail("Expected ValueError was not raised")
                except ValueError as exc:
                    err_msg = str(exc)
                    self.assertNotIn(sensitive_key, err_msg)
                    self.assertNotIn(sensitive_val, err_msg)
                    self.assertNotIn("trailing_secret", err_msg)
                    self.assertNotIn("export", err_msg)
                    # Verify message matches generic pattern
                    self.assertTrue(err_msg.startswith("Invalid environment configuration"))

    def test_inherently_atomic_no_partial_result(self):
        """A failure anywhere in text results in exception, no partial parsing."""
        text = "VALID_ONE=first\nBAD KEY=broken\nVALID_TWO=third"
        with self.assertRaises(ValueError):
            parse_environment(text)


class TestLoadEnvironmentFile(unittest.TestCase):
    """Test suite for load_environment_file production loader."""

    def test_default_missing_file_succeeds_without_mutation(self):
        """Default path (.env in cwd) when missing returns None and does not error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                env: dict[str, str] = {}
                result = load_environment_file(environment=env)
                self.assertIsNone(result)
                self.assertEqual(env, {})
            finally:
                os.chdir(orig_cwd)

    def test_default_existing_file_loads_successfully(self):
        """Default path (.env in cwd) when present loads successfully."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                env_file = Path(tmp_dir) / ".env"
                env_file.write_text("DEFAULT_KEY=default_val\n", encoding="utf-8")
                env: dict[str, str] = {}
                result = load_environment_file(environment=env)
                self.assertEqual(result, env_file)
                self.assertEqual(env, {"DEFAULT_KEY": "default_val"})
            finally:
                os.chdir(orig_cwd)

    def test_explicit_missing_file_raises_error(self):
        """Explicit missing path raises EnvironmentFileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_path = Path(tmp_dir) / "missing_config.txt"
            env: dict[str, str] = {}
            with self.assertRaises(EnvironmentFileNotFoundError) as ctx:
                load_environment_file(path=missing_path, environment=env)
            self.assertIsInstance(ctx.exception, FileNotFoundError)
            self.assertIsInstance(ctx.exception, ValueError)
            self.assertEqual(env, {})

    def test_explicit_missing_file_with_required_false_succeeds(self):
        """Explicit missing path with required=False returns None without error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_path = Path(tmp_dir) / "missing_config.txt"
            env: dict[str, str] = {}
            result = load_environment_file(path=missing_path, environment=env, required=False)
            self.assertIsNone(result)
            self.assertEqual(env, {})

    def test_default_missing_with_required_true_raises_error(self):
        """Default path with required=True raises error when absent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                env: dict[str, str] = {}
                with self.assertRaises(EnvironmentFileNotFoundError):
                    load_environment_file(path=None, environment=env, required=True)
                self.assertEqual(env, {})
            finally:
                os.chdir(orig_cwd)

    def test_precedence_existing_environment_wins_including_empty(self):
        """Existing process env wins over .env values, including empty string."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "testconfig.txt"
            config_file.write_text(
                "EXISTING=new_val\n"
                "EMPTY_VAR=from_file\n"
                "NEW_VAR=hello\n",
                encoding="utf-8",
            )
            env: dict[str, str] = {"EXISTING": "original", "EMPTY_VAR": ""}
            result = load_environment_file(path=config_file, environment=env)
            self.assertEqual(result, config_file)
            self.assertEqual(env["EXISTING"], "original")
            self.assertEqual(env["EMPTY_VAR"], "")
            self.assertEqual(env["NEW_VAR"], "hello")

    def test_quotes_and_literal_values_loaded(self):
        """Quoted and literal values are loaded without interpolation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "testconfig.txt"
            config_file.write_text(
                'DOUBLE="spaced value"\n'
                "SINGLE='single space'\n"
                "HASH=val#literal\n"
                "DOLLAR=$RAW_DOLLAR\n",
                encoding="utf-8",
            )
            env: dict[str, str] = {}
            load_environment_file(path=config_file, environment=env)
            self.assertEqual(
                env,
                {
                    "DOUBLE": "spaced value",
                    "SINGLE": "single space",
                    "HASH": "val#literal",
                    "DOLLAR": "$RAW_DOLLAR",
                },
            )

    def test_duplicate_key_fails_atomically_zero_mutation(self):
        """Duplicate keys in file raise ValueError with 0 mutation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "testconfig.txt"
            config_file.write_text(
                "VALID_FIRST=first\n"
                "DUP_KEY=val1\n"
                "DUP_KEY=val2\n",
                encoding="utf-8",
            )
            env: dict[str, str] = {"PRE_EXISTING": "untouched"}
            with self.assertRaises(ValueError):
                load_environment_file(path=config_file, environment=env)
            self.assertEqual(env, {"PRE_EXISTING": "untouched"})

    def test_invalid_syntax_fails_atomically_zero_mutation(self):
        """Syntax errors in file raise ValueError with 0 mutation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "testconfig.txt"
            config_file.write_text(
                "VALID_FIRST=first\n"
                "BAD LINE NO EQUALS\n",
                encoding="utf-8",
            )
            env: dict[str, str] = {"PRE_EXISTING": "untouched"}
            with self.assertRaises(ValueError):
                load_environment_file(path=config_file, environment=env)
            self.assertEqual(env, {"PRE_EXISTING": "untouched"})

    def test_invalid_utf8_fails_atomically_zero_mutation(self):
        """Invalid UTF-8 bytes raise ValueError with 0 mutation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "testconfig.txt"
            config_file.write_bytes(b"VALID_FIRST=first\nKEY=\xff\xfe\xfa\n")
            env: dict[str, str] = {"PRE_EXISTING": "untouched"}
            with self.assertRaises(ValueError):
                load_environment_file(path=config_file, environment=env)
            self.assertEqual(env, {"PRE_EXISTING": "untouched"})

    def test_file_size_boundary_64kib(self):
        """Files up to 64 KiB succeed; files exceeding 64 KiB raise ValueError with 0 mutation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Exactly 64 KiB (65536 bytes)
            # Create a line "KEY=" then padded with "a" to reach 65536 total bytes
            exact_file = Path(tmp_dir) / "exact64k.txt"
            prefix = "HUGE_VAL="
            padding = "a" * (65536 - len(prefix) - 1) + "\n"
            content_64k = (prefix + padding).encode("utf-8")
            self.assertEqual(len(content_64k), 65536)
            exact_file.write_bytes(content_64k)

            env: dict[str, str] = {}
            res = load_environment_file(path=exact_file, environment=env)
            self.assertEqual(res, exact_file)
            self.assertIn("HUGE_VAL", env)

            # 64 KiB + 1 byte (65537 bytes)
            over_file = Path(tmp_dir) / "over64k.txt"
            over_file.write_bytes(content_64k + b"x")
            self.assertEqual(len(content_64k + b"x"), 65537)

            env_over: dict[str, str] = {"PRE": "safe"}
            with self.assertRaises(ValueError):
                load_environment_file(path=over_file, environment=env_over)
            self.assertEqual(env_over, {"PRE": "safe"})

    def test_reject_directory(self):
        """Directory path is rejected with ValueError and 0 mutation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env: dict[str, str] = {"PRE": "safe"}
            with self.assertRaises(ValueError):
                load_environment_file(path=Path(tmp_dir), environment=env)
            self.assertEqual(env, {"PRE": "safe"})

    def test_reject_fifo_without_blocking(self):
        """FIFO path is rejected with ValueError without blocking."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fifo_path = Path(tmp_dir) / "test_fifo.pipe"
            os.mkfifo(fifo_path)
            env: dict[str, str] = {"PRE": "safe"}
            with self.assertRaises(ValueError):
                load_environment_file(path=fifo_path, environment=env)
            self.assertEqual(env, {"PRE": "safe"})

    def test_reject_symlink(self):
        """Symlink to regular file is rejected via O_NOFOLLOW with 0 mutation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_file = Path(tmp_dir) / "target.txt"
            target_file.write_text("FOO=bar\n", encoding="utf-8")
            link_path = Path(tmp_dir) / "link.env"
            os.symlink(target_file, link_path)

            env: dict[str, str] = {"PRE": "safe"}
            with self.assertRaises(ValueError):
                load_environment_file(path=link_path, environment=env)
            self.assertEqual(env, {"PRE": "safe"})

    def test_reject_broken_symlink(self):
        """Broken symlink is rejected with ValueError or FileNotFoundError and 0 mutation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            link_path = Path(tmp_dir) / "broken.env"
            os.symlink(Path(tmp_dir) / "nonexistent.txt", link_path)

            env: dict[str, str] = {"PRE": "safe"}
            with self.assertRaises(Exception) as ctx:
                load_environment_file(path=link_path, environment=env)
            self.assertTrue(isinstance(ctx.exception, (ValueError, FileNotFoundError)))
            self.assertEqual(env, {"PRE": "safe"})

    def test_error_chain_masking_and_sanitization(self):
        """Errors must not leak secret path/content and must strip __cause__ and __context__."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            secret_filename = "classified_secret_token.env"
            secret_value = "very_confidential_secret_value_12345"
            secret_file = Path(tmp_dir) / secret_filename
            # Invalid UTF-8 containing secret text
            secret_file.write_bytes(f"SECRET={secret_value}\xff\xfe".encode("latin1"))

            env: dict[str, str] = {}
            with self.assertRaises(ValueError) as ctx:
                load_environment_file(path=secret_file, environment=env)

            err = ctx.exception
            # Verify causes and contexts are stripped
            self.assertIsNone(err.__cause__)
            self.assertIsNone(err.__context__)

            # Verify no secret values or sensitive filenames leak
            err_msg = str(err)
            self.assertNotIn(secret_filename, err_msg)
            self.assertNotIn(secret_value, err_msg)


class TestCommandLineEnvironmentIntegration(unittest.TestCase):
    """Test suite for CLI --env-file integration and help skipping."""

    def test_cli_help_exits_without_reading_env(self):
        """CLI --help exits with code 0 without reading or validating .env in cwd."""
        from second_brain.__main__ import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                # Put a broken .env in cwd that would fail if read
                Path(".env").write_text("INVALID BROKEN ENV FILE\n", encoding="utf-8")
                with self.assertRaises(SystemExit) as ctx:
                    with patch("sys.stdout"):
                        main(["--help"])
                self.assertEqual(ctx.exception.code, 0)
            finally:
                os.chdir(orig_cwd)

    def test_cli_telegram_help_exits_without_reading_env(self):
        """CLI telegram --help exits with code 0 without reading .env."""
        from second_brain.__main__ import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                Path(".env").write_text("INVALID BROKEN ENV FILE\n", encoding="utf-8")
                with self.assertRaises(SystemExit) as ctx:
                    with patch("sys.stdout"):
                        main(["telegram", "--help"])
                self.assertEqual(ctx.exception.code, 0)
            finally:
                os.chdir(orig_cwd)

    def test_cli_explicit_missing_env_file_calls_parser_error(self):
        """CLI with explicit missing --env-file triggers parser.error (exit code 2) sanitized."""
        from second_brain.__main__ import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "super_secret_file.env"
            with self.assertRaises(SystemExit) as ctx:
                with patch("sys.stderr"):
                    main(["--env-file", str(missing), "--db", str(Path(tmp_dir) / "test.db"), "note", "list"])
            self.assertEqual(ctx.exception.code, 2)

    def test_cli_valid_env_file_loads_before_run(self):
        """CLI with valid --env-file loads environment variables before run."""
        from second_brain.__main__ import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Path(tmp_dir) / "testconfig.txt"
            config.write_text("CLI_TEST_LOADED_KEY=loaded_success\n", encoding="utf-8")
            db_path = Path(tmp_dir) / "test.db"

            test_env_key = "CLI_TEST_LOADED_KEY"
            if test_env_key in os.environ:
                del os.environ[test_env_key]

            try:
                with patch("sys.stdout"):
                    code = main([
                        "--env-file", str(config),
                        "--db", str(db_path),
                        "note", "save", "test note",
                        "--message-id", "msg-123",
                    ])
                self.assertEqual(code, 0)
                self.assertEqual(os.environ.get(test_env_key), "loaded_success")
            finally:
                if test_env_key in os.environ:
                    del os.environ[test_env_key]


if __name__ == "__main__":
    unittest.main()

