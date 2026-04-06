from ulid import ULID


def generate_id(seed_time: int | None = None) -> str:
    """Generate a new ULID. If seed_time (ms) provided, use it."""
    if seed_time is not None:
        return str(ULID.from_timestamp(seed_time / 1000))
    return str(ULID())


def extract_timestamp(ulid_str: str) -> int:
    """Extract ms timestamp from ULID."""
    return int(ULID.from_str(ulid_str).timestamp * 1000)


def compare_ulids(a: str, b: str) -> int:
    """Compare two ULIDs. Returns -1, 0, or 1."""
    ulid_a = ULID.from_str(a)
    ulid_b = ULID.from_str(b)
    if ulid_a < ulid_b:
        return -1
    elif ulid_a > ulid_b:
        return 1
    return 0
