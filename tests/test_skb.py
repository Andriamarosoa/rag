from app.agents.skb_search_agent import SkbSearchAgent
from app.skb.client import SkbClient


def test_skb_module_candidates_are_discovered_from_navigation_and_module_routes():
    html = """
    <html><head><title>SKB</title></head><body>
      <a href="/">Home</a>
      <a href="/modules/payroll">Payroll</a>
      <a href="/modules/hr">Human Resources</a>
      <a href="/pms">PMS</a>
      <script>{"moduleName":"ESS"}</script>
    </body></html>
    """

    modules = SkbClient._module_candidates_from_html(html, "http://skb.uniconsults.mu/")

    assert "Payroll" in modules
    assert "Human Resources" in modules
    assert "PMS" in modules
    assert "ESS" in modules
    assert "Home" not in modules


def test_search_agent_exposes_modules_in_input_schema_for_prompt_injection():
    client = SkbClient("http://skb.uniconsults.mu/")
    agent = SkbSearchAgent(client)
    try:
        agent.set_modules(["Payroll", "HR", "Payroll"])

        module_schema = agent.spec.input_schema["properties"]["module"]
        assert agent.modules == ["Payroll", "HR"]
        assert module_schema["enum"] == ["Payroll", "HR"]
        assert "Payroll" in agent.spec.description
        assert "HR" in agent.spec.description
        assert agent.spec.requires_confirmation is False
        assert agent.spec.write_action is False
    finally:
        import asyncio

        asyncio.run(client.close())
