from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")

def main():
    # Testing OpenAi
    llm_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    response_openai = llm_openai.invoke("Say 'setup complete!' in one word with an emoji")
    print(f"Response from OpenAi: {response_openai}")

    # Testing Athropic
    llm_antropic = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    response_anthropic = llm_antropic.invoke("Say 'setup complete!' in one word with an emoji")
    print(f"Response from Anthropic: {response_anthropic}")

if __name__ == "__main__":
    main()
