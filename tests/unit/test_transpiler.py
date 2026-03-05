"""PyMC -> SafeStan transpiler tests (all LLM calls mocked)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

from ppl_synthesis_reward_hacking.backends.transpiler import (
    TranspileResult,
    _extract_stan_code,
    check_transpiled_safety,
    transpile_batch,
    transpile_pymc_to_safestan,
)


def _make_mock_litellm(response_content: str) -> MagicMock:
    mock_mod = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_content
    mock_mod.completion.return_value = mock_response
    return mock_mod


class TestExtractStanCode:
    def test_from_stan_block(self):
        content = "Here is the code:\n```stan\ndata { int N; }\nmodel { }\n```\n"
        result = _extract_stan_code(content)
        assert result == "data { int N; }\nmodel { }"

    def test_from_code_block(self):
        content = "```code\ndata { int N; }\nparameters { real mu; }\nmodel { }\n```"
        result = _extract_stan_code(content)
        assert result == "data { int N; }\nparameters { real mu; }\nmodel { }"

    def test_stan_block_casefold(self):
        content = "```Stan\ndata { int N; }\nmodel { }\n```"
        result = _extract_stan_code(content)
        assert result == "data { int N; }\nmodel { }"

    def test_labeled_stan_body(self):
        content = "```python\ndata { int N; }\nmodel { }\n```"
        result = _extract_stan_code(content)
        assert result == "data { int N; }\nmodel { }"

    def test_from_bare_block_with_stan_keywords(self):
        content = "```\ndata { int N; }\nmodel { N ~ poisson(1); }\n```"
        result = _extract_stan_code(content)
        assert result is not None
        assert "data {" in result
        assert "model {" in result

    def test_labeled_nonstan_none(self):
        content = "```python\nprint('hello world')\n```"
        result = _extract_stan_code(content)
        assert result is None

    def test_returns_none_for_non_stan_bare_block(self):
        content = "```\nprint('hello world')\nx = 42\n```"
        result = _extract_stan_code(content)
        assert result is None

    def test_returns_none_for_no_block(self):
        content = "This is just plain text with no code blocks at all."
        result = _extract_stan_code(content)
        assert result is None


class TestTranspile:
    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
        result = transpile_pymc_to_safestan("import pymc as pm")
        assert not result.success
        assert result.stan_code is None
        assert "Missing API key" in result.error

    def test_with_mock_litellm(self, monkeypatch):
        monkeypatch.setenv("ZHIPUAI_API_KEY", "test-key")
        stan_response = (
            "```stan\n"
            "data { int<lower=0> N; array[N] int y; }\n"
            "parameters { real<lower=0,upper=1> p; }\n"
            "model { p ~ beta(1,1); y ~ bernoulli(p); }\n"
            "```"
        )
        mock_litellm = _make_mock_litellm(stan_response)
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

        result = transpile_pymc_to_safestan("import pymc as pm")
        assert result.success
        assert result.stan_code is not None
        assert "model {" in result.stan_code
        assert result.model_name == "anthropic/glm-5"

    def test_api_failure(self, monkeypatch):
        monkeypatch.setenv("ZHIPUAI_API_KEY", "test-key")
        mock_litellm = MagicMock()
        mock_litellm.completion.side_effect = ConnectionError("timeout")
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

        result = transpile_pymc_to_safestan("import pymc as pm")
        assert not result.success
        assert result.stan_code is None
        assert "LLM call failed" in result.error

    def test_no_stan_in_response(self, monkeypatch):
        monkeypatch.setenv("ZHIPUAI_API_KEY", "test-key")
        mock_litellm = _make_mock_litellm("I cannot translate this program because it has errors.")
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

        result = transpile_pymc_to_safestan("import pymc as pm")
        assert not result.success
        assert "No Stan code block" in result.error
        assert result.raw_response is not None

    def test_model_can_be_configured_via_env(self, monkeypatch):
        monkeypatch.setenv("PSRH_TRANSPILER_MODEL", "openai/gpt-5.4")
        monkeypatch.setenv("PSRH_TRANSPILER_API_KEY_ENV", "OPENAI_API_KEY")
        monkeypatch.setenv("PSRH_TRANSPILER_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("PSRH_TRANSPILER_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

        stan_response = "```stan\ndata { int N; }\nmodel { }\n```"
        mock_litellm = _make_mock_litellm(stan_response)
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

        result = transpile_pymc_to_safestan("import pymc as pm")
        assert result.success
        assert result.model_name == "openai/gpt-5.4"

        call_kwargs = mock_litellm.completion.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-5.4"
        assert call_kwargs["api_base"] == "https://api.openai.com/v1"
        assert call_kwargs["custom_llm_provider"] == "openai"
        assert call_kwargs["api_key"] == "test-openai-key"

    def test_explicit_model_overrides_env_model(self, monkeypatch):
        monkeypatch.setenv("PSRH_TRANSPILER_MODEL", "anthropic/glm-4.7")
        monkeypatch.setenv("ZHIPUAI_API_KEY", "test-key")
        stan_response = "```stan\ndata { int N; }\nmodel { }\n```"
        mock_litellm = _make_mock_litellm(stan_response)
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

        result = transpile_pymc_to_safestan(
            "import pymc as pm",
            model="anthropic/glm-5",
        )
        assert result.success
        assert result.model_name == "anthropic/glm-5"
        assert mock_litellm.completion.call_args.kwargs["model"] == "anthropic/glm-5"


class TestCheckTranspiledSafety:
    def test_honest_stan_passes(self):
        stan_code = """
data {
  int<lower=0> d;
  array[d] int<lower=0, upper=1> y;
}
parameters {
  real<lower=0, upper=1> p;
}
model {
  p ~ beta(1, 1);
  y ~ bernoulli(p);
}
"""
        result = TranspileResult(
            stan_code=stan_code,
            success=True,
            model_name="test",
        )
        accepted, reasons = check_transpiled_safety(result)
        assert accepted
        assert reasons == []

    def test_rejects_target_update(self):
        stan_code = """
data {
  int<lower=0> d;
  array[d] int<lower=0, upper=1> y;
}
parameters {
  real<lower=0, upper=1> p;
}
model {
  p ~ beta(1, 1);
  y ~ bernoulli(p);
  target += 10.0;
}
"""
        result = TranspileResult(
            stan_code=stan_code,
            success=True,
            model_name="test",
        )
        accepted, reasons = check_transpiled_safety(result)
        assert not accepted
        assert any("target updates" in r for r in reasons)

    def test_on_failed_result(self):
        result = TranspileResult(
            stan_code=None,
            success=False,
            model_name="test",
            error="some error",
        )
        accepted, reasons = check_transpiled_safety(result)
        assert not accepted
        assert reasons == ["transpilation failed"]


class TestTranspileBatch:
    def test_batch_returns_correct_count(self, monkeypatch):
        monkeypatch.setenv("ZHIPUAI_API_KEY", "test-key")
        stan_response = "```stan\ndata { int N; }\nmodel { }\n```"
        mock_litellm = _make_mock_litellm(stan_response)
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

        programs = ["prog1", "prog2", "prog3"]
        results = transpile_batch(programs)
        assert len(results) == 3
        assert all(r.success for r in results)
        assert mock_litellm.completion.call_count == 3


class TestPromptSelection:
    def test_strict_vs_relaxed_prompt(self, monkeypatch):
        monkeypatch.setenv("ZHIPUAI_API_KEY", "test-key")
        stan_response = "```stan\ndata { int N; }\nmodel { }\n```"
        mock_litellm = _make_mock_litellm(stan_response)
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

        transpile_pymc_to_safestan("code1", strict=True)
        strict_call = mock_litellm.completion.call_args_list[0]
        strict_msg = strict_call.kwargs["messages"][0]["content"]

        mock_litellm.reset_mock()
        mock_litellm.completion.return_value = _make_mock_litellm(
            stan_response
        ).completion.return_value

        transpile_pymc_to_safestan("code2", strict=False)
        relaxed_call = mock_litellm.completion.call_args_list[0]
        relaxed_msg = relaxed_call.kwargs["messages"][0]["content"]

        assert strict_msg != relaxed_msg
        assert "faithfully" in strict_msg.lower()
        assert "Do NOT fix bugs" in strict_msg
        assert "idiomatic" in relaxed_msg.lower()
