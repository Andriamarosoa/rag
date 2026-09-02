from app.ollama.client import OllamaNativeClient


def test_native_ollama_base_url_strips_openai_v1_suffix():
    assert (
        OllamaNativeClient.normalize_base_url("http://100.89.128.87:11434/v1")
        == "http://100.89.128.87:11434"
    )


def test_native_ollama_base_url_keeps_native_root():
    assert (
        OllamaNativeClient.normalize_base_url("http://100.89.128.87:11434/")
        == "http://100.89.128.87:11434"
    )
