from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

def demo_basic_chain():
  """Demonstrates a basic chain using LCEL and runnables."""
  prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer in one sentence: {question}")
  model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
  parser = StrOutputParser()

  # Compose with pipe operator
  chain = prompt | model | parser

  # Execute the chain with an input
  result = chain.invoke({"question": "What is LangChain?"})

  print(f"Response: {result}")

  return chain

def demo_batch_execution():
  """Demonstrate batch execution for multiple inputs."""
  prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
  model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
  parser = StrOutputParser()

  chain = prompt | model | parser

  # Batch - run with multiple inputs
  inputs = [
    {"text": "Hello, how are you?"},
    {"text": "What is your name?"},
    {"text": "Where is the nearest restaurant?"}
    ]
  results = chain.batch(inputs)

  for text in zip(inputs, results, strict=True):
    print(f"Input: {text[0]["text"]} => Output: {text[1]}")

def demo_streaming():
  """Demonstrate streaming for real-time output."""
  prompt = ChatPromptTemplate.from_template("Write a 300 words story about: {topic}")
  model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
  parser = StrOutputParser()

  chain = prompt | model | parser

  # Streaming - run sith streaming enabled
  print("Streaming output: ")
  for chunk in chain.stream({"topic": "Medellîn"}):
    print(chunk, end="",  flush=True)
  print()

def demo_schema_inspection():
  """Demonstrate input/outpout schema inspection"""
  prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
  model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
  parser = StrOutputParser()

  chain = prompt | model | parser

  # Inspect input and output schemas
  input_schema = chain.input_schema.model_json_schema()
  output_schema = chain.output_schema.model_json_schema()

  print(f"Input schema: {input_schema}")
  print(f"Output schema: {output_schema}")

def exercise_first_chain():
  prompt = ChatPromptTemplate.from_template("You are a marketing expert. Based on a product and an audience, you should generate a product tagline. Product: {product}, Audience: {audience}")
  model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
  parser = StrOutputParser()

  chain = prompt | model | parser

  result = chain.invoke({"product": "Ai Course", "audience": "developers"})

  print(f"Response: {result}")

def new_way():
  # The universal way to initialize a model
  model = init_chat_model("openai:gpt-4o-mini", temperature=0.7, max_tokens=500)



if __name__ == "__main__":
  # demo_basic_chain()
  # demo_batch_execution()
  # demo_streaming()
  # demo_schema_inspection()
  exercise_first_chain()