"""Unit tests for src/util/atomic_io.py (atomic_write)."""

import os

import pytest

from src.util import atomic_write


class TestAtomicWrite:
    def test_creates_file_with_written_content(self, tmp_path):
        target = tmp_path / "data.txt"
        with atomic_write(str(target)) as tmp:
            with open(tmp, "w") as f:
                f.write("hello")
        assert target.read_text() == "hello"

    def test_replaces_existing_file(self, tmp_path):
        target = tmp_path / "data.txt"
        target.write_text("old")
        with atomic_write(str(target)) as tmp:
            with open(tmp, "w") as f:
                f.write("new")
        assert target.read_text() == "new"

    def test_failure_preserves_existing_file(self, tmp_path):
        target = tmp_path / "data.txt"
        target.write_text("old")
        with pytest.raises(RuntimeError):
            with atomic_write(str(target)) as tmp:
                with open(tmp, "w") as f:
                    f.write("partial garbage")
                raise RuntimeError("simulated crash mid-write")
        assert target.read_text() == "old"

    def test_failure_leaves_no_temp_files(self, tmp_path):
        target = tmp_path / "data.txt"
        with pytest.raises(RuntimeError):
            with atomic_write(str(target)):
                raise RuntimeError("simulated crash")
        assert os.listdir(tmp_path) == []

    def test_success_leaves_no_temp_files(self, tmp_path):
        target = tmp_path / "data.txt"
        with atomic_write(str(target)) as tmp:
            with open(tmp, "w") as f:
                f.write("x")
        assert os.listdir(tmp_path) == ["data.txt"]

    def test_temp_file_in_same_directory_as_target(self, tmp_path):
        # Same-directory temp files guarantee os.replace is a same-filesystem
        # rename, which is what makes the write atomic.
        target = tmp_path / "data.txt"
        with atomic_write(str(target)) as tmp:
            assert os.path.dirname(tmp) == str(tmp_path)
            with open(tmp, "w") as f:
                f.write("x")

    def test_relative_path_without_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with atomic_write("data.txt") as tmp:
            with open(tmp, "w") as f:
                f.write("relative")
        assert (tmp_path / "data.txt").read_text() == "relative"
