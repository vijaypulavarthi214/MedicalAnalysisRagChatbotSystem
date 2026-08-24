from app.config import get_settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("HYBRID_ALPHA", "0.5")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.hybrid_alpha == 0.5
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    get_settings.cache_clear()


def test_cors_origin_list_parses_csv(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.com, https://b.com")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.cors_origin_list == ["https://a.com", "https://b.com"]
    get_settings.cache_clear()


def test_max_file_size_bytes(monkeypatch):
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "5")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.max_file_size_bytes == 5 * 1024 * 1024
    get_settings.cache_clear()
