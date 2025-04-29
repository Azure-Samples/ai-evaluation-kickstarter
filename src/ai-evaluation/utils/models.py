import os
from typing import TypedDict, Any, Optional, List, Dict
from utils.util import load_dotenv_from_azd

# Import Semantic Kernel
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory

# Import Azure Identity
from azure.identity.aio import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.aio import ChatCompletionsClient

# Import the telemetry module
from utils.telemetry.sk_tracing import setup_semantic_kernel_tracing

# Load environment variables
load_dotenv_from_azd()

# Update to read supported models from an environment variable
SUPPORTED_MODELS = [
    model.strip() for model in os.getenv("SUPPORTED_MODELS", "gpt-4o,gpt-4o-mini").split(',')
]

class ModelEndpoints:
    """Class to handle calls to various model endpoints"""
    
    def __init__(self, model_name: str):
        """
        Initialize with the selected model name.
        
        Args:
            model_name: Name of the model to use for evaluation
        """
        self.model_name = model_name
        self.set_system_message("You are a helpful assistant.")
        
        # Load environment variables
        load_dotenv(override=True)
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        credential = AzureKeyCredential(azure_api_key) if azure_api_key else DefaultAzureCredential()

        # Initialize the kernel
        self.kernel = Kernel()
        
        # Use the extracted telemetry setup function
        setup_semantic_kernel_tracing()
        
        # Create the chat completion service, with workaround for gpt-4.1
        # Apply workaround: use AzureChatCompletion for gpt-4.1 models
        # Use the Azure OpenAI endpoint with deployment_name instead of inference API
        self.chat_completion_service = AzureAIInferenceChatCompletion(
            ai_model_id=model_name,
            client=ChatCompletionsClient(
                endpoint=f"{str(endpoint).strip('/')}/openai/deployments/{model_name}",
                credential=credential,
                credential_scopes=["https://cognitiveservices.azure.com/.default"]
            ),
            service_id="ai-evaluation-service"
        )
        
        # Add the chat completion service to the kernel
        self.kernel.add_service(self.chat_completion_service)

    class Response(TypedDict):
        query: str
        response: str
        context: str
    
    @staticmethod
    def get_supported_models() -> List[str]:
        """
        Returns a list of supported model names.
        
        Returns:
            List of supported model names
        """
        return SUPPORTED_MODELS
    
    def set_system_message(self, system_message: str):
        """
        Set a custom system message for the model.

        Args:
            system_message: The system message to set.
        """
        self.system_message = system_message

    async def __call__(self, query: str, context: Optional[str] = None) -> Response:
        """
        Call the appropriate model with the query and context.
        
        Args:
            query: The user query
            context: Optional context information
            
        Returns:
            Dict with query, response, and other info
        """
        response = ""
        prompt = query
        
        # Include context if provided
        if context:
            prompt = f"## Context: {context}\n## Question: {query}"
        
        # Call the model using Semantic Kernel
        response = await self._call_semantic_kernel(prompt)
            
        # Return in the format expected by the evaluation SDK
        return {
            "system_message": self.system_message,
            "query": query,
            "response": response,
            "context": context or "",
        }
    
    async def _call_semantic_kernel(self, prompt: str) -> str:
        """Call Azure OpenAI API using Semantic Kernel and return the response content"""
        try:
            print("---------------------")
            print("Using Semantic Kernel")
            print(f"Using deployment name: {self.model_name}")
            print(f"Using system message: {self.system_message}")
            print(f"Using query: {prompt}")
            
            # Create a chat history with the system message and user prompt
            chat_history = ChatHistory()
            chat_history.add_system_message(self.system_message)
            chat_history.add_user_message(prompt)
            
            # Get the chat completion service from the kernel
            chat_completion_service = self.kernel.get_service(service_id="ai-evaluation-service")
            
            # Get completion settings
            settings = self.kernel.get_prompt_execution_settings_from_service_id("ai-evaluation-service")
            settings.temperature = 0.7
            settings.max_tokens = 800
            settings.top_p = 0.95
            
            # Complete the prompt with the chat completion service
            response = await chat_completion_service.get_chat_message_content(
                chat_history=chat_history,
                settings=settings
            )
            
            if response:
                print(f">> Response: {response}")
                print("---------------------")

                # Add the chat message to the chat history to keep track of the conversation.
                chat_history.add_message(response)
                return response.content if hasattr(response, 'content') else str(response)
            else:
                print("No response received from the model.")
                return "No response received from the model."
        except Exception as e:
            print(f"Error calling Azure OpenAI with Semantic Kernel: {str(e)}")
            return f"Error calling Azure OpenAI with Semantic Kernel: {str(e)}"