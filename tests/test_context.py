from app.context.manager import ContextManager


def test_token_estimate_is_nonzero():
    assert ContextManager.estimate_tokens("hello") >= 1
    assert ContextManager.estimate_tokens("a" * 400) == 100
