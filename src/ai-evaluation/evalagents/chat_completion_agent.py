# Copyright (c) Microsoft. All rights reserved.
from utils.util import load_dotenv_from_azd
import os

import asyncio
from typing import Annotated, Literal

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread, AgentResponseItem
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.contents import AuthorRole, TextContent

from utils.models import ModelEndpoints

"""
The following sample demonstrates how to create a chat completion agent that
answers questions about a sample menu using a Semantic Kernel Plugin. The Chat
Completion Service is passed directly via the ChatCompletionAgent constructor.
Additionally, the plugin is supplied via the constructor.
"""
# Load environment variables
load_dotenv_from_azd()

# Define a plugin with tools for the agent to use
class MenuPlugin:
    """A sample Menu Plugin used for the concept sample."""

    @kernel_function(description="Provides a list of specials from the menu.")
    def get_specials(self) -> Annotated[str, "Returns the specials from the menu."]:
        return """
        Special Soup: Clam Chowder
        Special Salad: Cobb Salad
        Special Drink: Chai Tea
        """

    @kernel_function(description="Provides the price of the requested menu item.")
    def get_item_price(
        self, menu_item: Annotated[str, "The name of the menu item."]
    ) -> Annotated[str, "Returns the price of the menu item."]:
        return "$9.99"


# Simulate a conversation with the agent
USER_INPUTS = [
    "Hello",
    "What is the special drink today?",
    "What does that cost?",
    "Thank you",
]

async def _run_semantic_kernel_agent(model_name: str, user_inputs: list) -> list:
    """
    Create and run the chat completion agent using the semantic-kernel framework.
    """

    modelEndpoint = ModelEndpoints(model_name=model_name)

    agent = ChatCompletionAgent(
        service=modelEndpoint.chat_completion_service,
        name="Chef",
        instructions="Answer questions about the menu.",
        plugins=[MenuPlugin()],
    )
   
    print(f"\n{80 * '='}")
    print("# Sample AI Agent Conversation begins...")

    thread: ChatHistoryAgentThread = None
    responses = []
    
    for user_input in user_inputs:
        response = await agent.get_response(messages=user_input, thread=thread)
        print(f"## User: {user_input}")
        print(f"## {response.name}: {response}")
        responses.append((user_input, response.name, str(response)))
        thread = response.thread

    print("# End of conversation")
    print(f"\n{80 * '='}")
    
    # Return the agent and thread along with responses, evaluation will be done externally
     
    # Optionally delete thread after external evaluation
    # if thread: await thread.delete()
    return agent, thread, responses


async def _run_openai_agent(model_name: str, user_inputs: list) -> list:
    """
    Handles running the chat agent using the OpenAI Agent SDK.
    """
    from agents import Agent, Runner, function_tool, set_default_openai_client, trace
    from openai import AsyncAzureOpenAI

    # Create OpenAI client using Azure OpenAI
    openai_client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-03-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=model_name
    )

    # Set the default OpenAI client for the Agents SDK
    set_default_openai_client(openai_client)

    # Register MenuPlugin methods as tools
    menu_plugin = MenuPlugin()
    get_specials_tool = function_tool(menu_plugin.get_specials)
    get_item_price_tool = function_tool(menu_plugin.get_item_price)

    agent = Agent(
        name="Chef",
        instructions="Answer questions about the menu.",
        tools=[get_specials_tool, get_item_price_tool],
    )

    responses = []
    for user_input in user_inputs:
        with trace("OpenAI Agent workflow"):
            response = await Runner.run(agent, input=user_input)
            print(f"# Raw response: {response.raw_responses}")
            print(f"# User: {user_input}")
            print(f"# {agent.name}: {response}")
        responses.append((user_input, agent.name, response.final_output))

    return responses

async def run_chat_agent(
    model_name: str = "gpt-4o",
    user_inputs: list = USER_INPUTS,
    agent_framework: Literal["semantic-kernel", "openai"] = "semantic-kernel"
) -> list:
    """
    Run the chat completion agent with the provided user inputs.

    Args:
        user_inputs (list): List of user input strings.
        agent_framework (Literal): The agent framework to use. Must be either 'semantic-kernel' or 'openai'.

    Returns:
        list: List of agent responses.
    """
    match agent_framework:
        case "semantic-kernel":
            # Return agent, thread, and responses for external evaluation
            return await _run_semantic_kernel_agent(model_name, user_inputs)
        case "openai":
            return await _run_openai_agent(model_name, user_inputs)
        case _:
            raise ValueError(f"Unknown agent_framework: {agent_framework}")


async def main():
    responses = await run_chat_agent(model_name="gpt-4o", user_inputs=USER_INPUTS)
    for user_input, agent_name, response in responses:
        print(f"# User: {user_input}")
        print(f"# {agent_name}: {response} ")

    """
    Sample output:
    # User: Hello
    # Host: Hello! How can I assist you today?
    # User: What is the special soup?
    # Host: The special soup is Clam Chowder.
    # User: What does that cost?
    # Host: The special soup, Clam Chowder, costs $9.99.
    # User: Thank you
    # Host: You're welcome! If you have any more questions, feel free to ask. Enjoy your day!
    """


if __name__ == "__main__":
    asyncio.run(main())