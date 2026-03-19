from langchain_groq import ChatGroq

model = ChatGroq(
    model="groq/compound",  # Free model
    api_key="GROQ_API_KEY"
)

response = model.invoke("Hello!")
print(response.content)
