from RAG_usage import RAGOrchestrator

from ollama_usage.OllamaAgent import Models, OllamaAgent

CITY_WEATHER: str = "Brescia"
RAG_QUESTION: str = "What are the overall performance of Answer Refinement?"

print(f"Hi. Let's see the weather at {CITY_WEATHER}")

agent = OllamaAgent(Models.LLAMA)

print(agent.askLLM(f"What is the wheater at {CITY_WEATHER}?"))

print("\n\nNow we can ask something at our RAG model. By default, it is a question related to my thesis.\nAsk questions based on the document placed in the documents folder!")

print(RAGOrchestrator.askRAG(RAG_QUESTION))