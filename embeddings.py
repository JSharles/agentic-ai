from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# single text
text = "This is a sample text to be embedded."
embedding = embeddings.embed_query(text)
print(f"Embedding for single text: {embedding}")

#  multiple texts
embeds = embeddings.embed_documents(
    ["This is the first document.", "This is the second document."]
)
print(f"Embeddings for multiple texts: {embeds}")
print(f"Number of embeddings returned: {len(embeds)} ")
print(f"Length of each embedding: {len(embeds[0])}")
