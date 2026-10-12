from langchain_ollama import ChatOllama


llm = ChatOllama(
    model = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:latest",
    temperature=0
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content_blocks)

print(ai_msg.content_blocks[0]["text"])