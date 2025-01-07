from trademarkpredictor.config import get_config
from trademarkpredictor.agent import TrademarkCaseAgent
import os

def test_environment():
    # Test configuration
    config = get_config()
    assert config["PROJECT_ID"] == "trademark-case-agent"
    print("✓ Configuration verified")

    # Test Pinecone settings
    assert os.getenv('PINECONE_API_KEY'), "Pinecone API key missing"
    print("✓ Pinecone configuration verified")

    # Test agent initialization
    agent = TrademarkCaseAgent()
    print("✓ Agent initialized successfully")

if __name__ == "__main__":
    test_environment()