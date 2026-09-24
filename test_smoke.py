"""Run after `python ingest.py`; verifies both graph routes in default mock mode."""
from main import chain
for query in ["What is the delivery fee?", "Tell me a joke."]:
    result=chain.invoke({"query":query}); print(query, result["intent"], result["response"].model_dump())
