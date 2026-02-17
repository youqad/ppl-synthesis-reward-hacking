from ppl_synthesis_reward_hacking.utils.hashing import normalized_text_hash, stable_hash


def test_stable_hash_order_independent() -> None:
    data_a = {"b": 2, "a": 1}
    data_b = {"a": 1, "b": 2}
    assert stable_hash(data_a) == stable_hash(data_b)


def test_normalized_text_hash_ignores_whitespace_differences() -> None:
    text_a = "def model(data):\n    return data\n"
    text_b = "def   model(data):    \nreturn   data"
    assert normalized_text_hash(text_a) == normalized_text_hash(text_b)
