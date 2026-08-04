import fakeredis
import pytest

from spamdet.outbreak.lsh import RedisLSHIndex


@pytest.fixture
def redis_client():
    return fakeredis.FakeStrictRedis(decode_responses=False)


def test_rejects_hash_bits_not_divisible_by_num_bands(redis_client):
    with pytest.raises(ValueError):
        RedisLSHIndex(redis_client, num_bands=5, hash_bits=64)


def test_add_and_get_fingerprint_round_trip(redis_client):
    index = RedisLSHIndex(redis_client, num_bands=4, hash_bits=64)
    index.add("msg-1", 0xDEADBEEF)
    assert index.get_fingerprint("msg-1") == 0xDEADBEEF


def test_get_fingerprint_returns_none_for_unknown_message(redis_client):
    index = RedisLSHIndex(redis_client, num_bands=4, hash_bits=64)
    assert index.get_fingerprint("nope") is None


def test_candidates_finds_message_sharing_a_band(redis_client):
    index = RedisLSHIndex(redis_client, num_bands=4, hash_bits=64)
    fp_a = 0b0000_1111_0000_1111_0000_1111_0000_1111_0000_1111_0000_1111_0000_1111_0000_1111
    fp_b_same_first_band = fp_a ^ (1 << 20)  # differs outside band 0 (bits 0-15)

    index.add("msg-a", fp_a)
    candidates = RedisLSHIndex(redis_client, num_bands=4, hash_bits=64).candidates(fp_b_same_first_band)
    assert "msg-a" in candidates


def test_candidates_empty_when_no_band_overlap(redis_client):
    index = RedisLSHIndex(redis_client, num_bands=4, hash_bits=64)
    index.add("msg-a", 0x0000_0000_0000_0000)
    candidates = index.candidates(0xFFFF_FFFF_FFFF_FFFF)
    assert candidates == set()


def test_add_with_ttl_sets_expiry(redis_client):
    index = RedisLSHIndex(redis_client, num_bands=4, hash_bits=64)
    index.add("msg-1", 12345, ttl_seconds=3600)
    ttl = redis_client.ttl(index._fingerprint_key("msg-1"))
    assert ttl > 0
