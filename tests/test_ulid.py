import re
import time

from iecp_core.utils import compare_ulids, extract_timestamp, generate_id


def test_generates_valid_ulid():
    ulid = generate_id()
    assert len(ulid) == 26
    assert re.match(r"^[0-9A-HJKMNP-TV-Z]{26}$", ulid)


def test_generates_unique_ids():
    ids = {generate_id() for _ in range(100)}
    assert len(ids) == 100


def test_lexicographically_sortable():
    earlier = generate_id(seed_time=1000000)
    later = generate_id(seed_time=2000000)
    assert earlier < later


def test_extracts_timestamp_correctly():
    seed_time = 1700000000000
    ulid = generate_id(seed_time=seed_time)
    extracted = extract_timestamp(ulid)
    assert abs(extracted - seed_time) < 2


def test_timestamp_matches_current_time():
    before = int(time.time() * 1000)
    ulid = generate_id()
    after = int(time.time() * 1000)
    extracted = extract_timestamp(ulid)
    assert before <= extracted <= after + 1


def test_compare_ulids_negative_when_a_less():
    a = generate_id(seed_time=1000000)
    b = generate_id(seed_time=2000000)
    assert compare_ulids(a, b) < 0


def test_compare_ulids_positive_when_a_greater():
    a = generate_id(seed_time=2000000)
    b = generate_id(seed_time=1000000)
    assert compare_ulids(a, b) > 0


def test_compare_ulids_zero_for_equal():
    ulid = generate_id()
    assert compare_ulids(ulid, ulid) == 0
