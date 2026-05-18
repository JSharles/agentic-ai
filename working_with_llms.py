from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def demo_init_chat_model():
  chat_model = init_chat_model(
    model="gpt-4o-mini",
    # model_provider="openai",
    temperature=0.7,
    streaming=True,
    max_retries=3
  )
  response = chat_model.invoke("What is the capital of France? Answer in one word.")
  print(f"Response form GPT: {response.content}")

  # Easy to switch model providers
  if(os.getenv("ANTHROPIC_API_KEY")):
    claude_chat_model = init_chat_model(
      model="claude-sonnet-4-5-20250929",
      model_provider="anthropic",
      temperature=0.7,
      streaming=True,
      max_retries=3
    )

    response = claude_chat_model.invoke("What is the capital of France? Answer in one word.")
    print(f"Response from Claude: {response.content}")

def demo_model_comparison():
    prompt = "Explain recursion in one sentence."

    models = {
        "gpt-4o-mini": init_chat_model(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=False,
        ),
        "gpt-4o": init_chat_model(
            model="gpt-4o",
            temperature=0.7,
            streaming=False,
        ),
    }

    # add anthropic model if available
    if os.getenv("ANTHROPIC_API_KEY"):
        models["claude-sonnet-4-5-20250929"] = init_chat_model(
            model="claude-sonnet-4-5-20250929",
            model_provider="anthropic",
            temperature=0.7,
            streaming=False,
        )

    print(f"Prompt: {prompt}\n")

    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n")

def demo_message():
   model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
   #Using message objects (more control over roles)

   messages = [
      SystemMessage(content="You are a pirate. Always answer like a pirate."),
      HumanMessage(content="What's the weather like today?")
   ]

  #  print("Using message objects:")
  #  print(f"Message: {messages[0]} | {messages[1]}")

   response = model.invoke(messages)
   print(f"Response using message objects: {response.content}")

   # Multi-turn conversation using message objects
   messages.append(response) # add model's response to the conversation
   messages.append(HumanMessage(content="What about tomorrow?"))

   print("\nMulti-turn conversation:")
   response = model.invoke(messages)
   print(f"Follow-up response from the Pirate: {response.content}")

def exercise_multi_model():

   def get_responses(question: str, model_names: list[str]) -> dict[str, str]:
      responses = {}
      for model_name in model_names:
         model = init_chat_model(
            model=model_name,
            temperature=0.7,
            streaming=False,
         )
         response = model.invoke(question)
         responses[model_name] = response.content
      return responses

   results = get_responses("What is AI?", ["gpt-4o-mini", "gpt-4o"])
   for model, answer in results.items():
      print(f"Response from {model}: {answer}\n")

  #  messages = [
  #     SystemMessage(content="You are a tech market expert, be concise."),
  #     HumanMessage(content="In 2026, in the tech ecosystem, what are the most in-demand skills fro developers?")
  #  ]
  #  models = {

  #       "gpt-4o": init_chat_model(
  #           model="gpt-4o",
  #           temperature=0.7,
  #           streaming=False,
  #       ),
  #        "claude-sonnet-4-5-20250929": init_chat_model(
  #           model="gpt-4o-mini",
  #           temperature=0.7,
  #           streaming=False,
  #       ),
  #   }

  #  for model_name, model in models.items():
  #     response = model.invoke(messages)
  #     print(f"Response from {model_name}: {response.content}")
  #     messages.append(response)

  #     messages.append(HumanMessage("Dans ce que tu cites, où se trouve la compétence consistant à implémenter l'IA dans un produit web, ou bien automatiser des flux business? Je pense a des outils comme langchain, langgraph ..etc. Quel valeur cela a t il ?"))
  #     response = model.invoke(messages)

  #     print(f"------------------------- Follow up {model_name} ------------------")
  #     print(f"{response.content}")


if __name__ == "__main__":
  # demo_init_chat_model()
  # demo_model_comparison()
  # demo_message()
  exercise_multi_model()