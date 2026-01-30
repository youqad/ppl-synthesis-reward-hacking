from ppl_synthesis_reward_hacking.utils.hashing import stable_hash


def test_stable_hash_order_independent() -> None:
    data_a = {"b": 2, "a": 1}
    data_b = {"a": 1, "b": 2}
    assert stable_hash(data_a) == stable_hash(data_b)
