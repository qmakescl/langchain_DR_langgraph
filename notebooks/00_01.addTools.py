# Load environment variables and set up auto-reload
from dotenv import load_dotenv
load_dotenv()


# Define the tool
from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=2)

# tool을 리스트로 구성
tools = [tool]


# Initialize chat model
import os
from langchain.chat_models import init_chat_model

llm = init_chat_model("google_genai:gemini-2.5-flash")


# tools with Graph
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Modification: tell the LLM which tools it can call : tool calling
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)


# create a function to run the tools
import json
from langchain_core.messages import ToolMessage

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )

            outputs.append(
                ToolMessage(
                    content=formatted_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


# tool_node는 '이런 기능을 할 수 있다'는 설계도와 실제 부품일 뿐,
# 아직 아무런 능력이 없습니다.
tool_node = BasicToolNode(tools=[tool])

# 바로 이 순간! LangGraph에게 tool_node를 등록하며 약속이 생겨납니다.
# 제어의 역전(IoC, Inversion of Control)
graph_builder.add_node("tools", tool_node)


# define the conditional_edges
def route_tools( state: State, ):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    # tool_calls 생성
    # tool_calls 속성은 chatbot 노드 내부에서 llm.bind_tools(tools)로 생성된 llm_with_tools가 호출될 때 만들어집니다. 
    # LLM(거대 언어 모델)이 사용자의 질문에 답변하기 위해 도구(Tool)를 사용해야 한다고 판단하면, 
    # 일반적인 텍스트 답변 대신 tool_calls 속성을 포함한 AIMessage 객체를 반환합니다. 
    # 이 객체에 바로 그 tool_calls가 들어있습니다.
    #LLM의 판단 및 tool_calls 생성:
    #   LLM은 사용자의 질문("What do you know about LangGraph?")을 분석합니다.
    #   LLM은 이 질문에 스스로 답변할 수 없으며, 웹 검색이 필요하다고 판단합니다.
    #   이때, bind_tools를 통해 알고 있던 TavilySearch 도구를 사용하기로 결정합니다.
    #   LLM은 일반적인 답변 대신, 어떤 도구를 어떤 인자(argument)로 호출해야 하는지에 대한 정보를 담은 특별한 AIMessage를 생성합니다. 이 정보가 바로 tool_calls 속성에 저장됩니다.
    #   tool_calls는 [{'name': 'tavily_search_results_json', 'args': {'query': 'LangGraph'}, 'id': '...'}, ...]와 같은 형태의 리스트가 됩니다.
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


# Ask the bot questions
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
