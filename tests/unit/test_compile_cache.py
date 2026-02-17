from __future__ import annotations

from pathlib import Path

from ppl_synthesis_reward_hacking.backends.stan.compile_cache import get_compile_dir


def test_get_compile_dir_returns_path():
    result = get_compile_dir(Path("/tmp/cache"), "abc123")
    assert isinstance(result, Path)


def test_get_compile_dir_nests_under_stan_subdir():
    result = get_compile_dir(Path("/tmp/cache"), "abc123")
    assert result == Path("/tmp/cache/stan/abc123")


def test_get_compile_dir_preserves_cache_root():
    root = Path("/custom/path/to/cache")
    result = get_compile_dir(root, "hash_value")
    assert result.parent.parent == root


def test_get_compile_dir_uses_model_hash_as_leaf():
    model_hash = "deadbeef1234"
    result = get_compile_dir(Path("/tmp"), model_hash)
    assert result.name == model_hash


def test_get_compile_dir_different_hashes_different_dirs():
    a = get_compile_dir(Path("/tmp/cache"), "hash_a")
    b = get_compile_dir(Path("/tmp/cache"), "hash_b")
    assert a != b


def test_get_compile_dir_same_inputs_same_output():
    a = get_compile_dir(Path("/tmp/cache"), "same_hash")
    b = get_compile_dir(Path("/tmp/cache"), "same_hash")
    assert a == b


def test_get_compile_dir_result_is_absolute_when_root_is_absolute():
    result = get_compile_dir(Path("/abs/root"), "h")
    assert result.is_absolute()


def test_get_compile_dir_result_is_relative_when_root_is_relative():
    result = get_compile_dir(Path("relative/root"), "h")
    assert not result.is_absolute()
