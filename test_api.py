from langchain_groq import ChatGroq

model = ChatGroq(
    model="groq/compound",  # Free model
    api_key="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"
)

response = model.invoke("Hello!")
print(response.content)