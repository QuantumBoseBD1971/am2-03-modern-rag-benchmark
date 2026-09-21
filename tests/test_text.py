from modern_rag_benchmark.text import tokenise


def test_tokenise_normalises_case_and_punctuation() -> None:
    assert tokenise("Lithium-Battery, Storage!") == [
        "lithium",
        "battery",
        "storage",
    ]
