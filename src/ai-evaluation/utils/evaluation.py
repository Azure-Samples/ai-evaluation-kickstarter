from typing import Any, Dict, List
import os
import random
import tempfile
import pandas as pd
from abc import ABC, abstractmethod
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.evaluation import (
    evaluate,
    AzureAIProject,
    AzureOpenAIModelConfiguration,
    EvaluatorConfig,
    GroundednessEvaluator,
    RetrievalEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator, 
    FluencyEvaluator,
    F1ScoreEvaluator,
    RougeScoreEvaluator,
    GleuScoreEvaluator,
    BleuScoreEvaluator,
    MeteorScoreEvaluator,
    ViolenceEvaluator,
    SexualEvaluator,
    SelfHarmEvaluator,
    HateUnfairnessEvaluator,
    IndirectAttackEvaluator,
    ProtectedMaterialEvaluator,
    QAEvaluator,
    ContentSafetyEvaluator,
    ToolCallAccuracyEvaluator,
    IntentResolutionEvaluator,
    TaskAdherenceEvaluator,
)
from utils.models import ModelEndpoints
from utils.util import load_dotenv_from_azd

# Load environment variables
load_dotenv_from_azd()

class ResponseDataset:
    """Dataset for pre-generated responses used in evaluation."""
    def __init__(self, data=None):
        """Initialize the ResponseDataset with optional data.
        Args:
            data: Optional initial data. Can be a pandas DataFrame, a dictionary, or a list of dictionaries.
        """
        if data is None:
            self.data = pd.DataFrame(columns=["query", "context", "response", "ground_truth", "system_message"])
        elif isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, dict):
            self.data = pd.DataFrame([data])
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            self.data = pd.DataFrame(data)
        else:
            raise ValueError("Data must be a pandas DataFrame, a dictionary, or a list of dictionaries")

    def add_response(self, query: str, response: str, context: str = "", ground_truth: str = "", system_message: str = ""):
        """Add a response entry to the dataset.
        Args:
            query: The input query.
            response: The model's response.
            context: Optional context for the query.
            ground_truth: Optional ground truth answer.
            system_message: Optional system message used for the response.
        """
        new_row = {
            "query": query,
            "response": response,
            "context": context,
            "ground_truth": ground_truth,
            "system_message": system_message
        }
        self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)

    async def generate_responses(self, model_endpoint: ModelEndpoints, queries: List[Dict]):
        """Generate responses for a list of queries using the provided model endpoint.
        Args:
            model_endpoint: The model endpoint to use for generating responses.
            queries: A list of dictionaries containing query information.
        """
        for query_dict in queries:
            query = query_dict.get("query", "")
            context = query_dict.get("context", "")
            ground_truth = query_dict.get("ground_truth", "")
            system_message = query_dict.get("system_message", "")
            if system_message:
                model_endpoint.set_system_message(system_message)
            result = await model_endpoint(query, context)
            self.add_response(
                query=query,
                response=result["response"],
                context=context,
                ground_truth=ground_truth,
                system_message=result["system_message"]
            )

    def to_jsonl(self, file_path: str = None):
        """Export the dataset to a JSONL file.
        Args:
            file_path: Optional file path to save the JSONL file. If None, a temporary file is created.
        Returns:
            The file path of the JSONL file.
        """
        if file_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as temp_file:
                self.data.to_json(temp_file.name, orient="records", lines=True)
                return temp_file.name
        self.data.to_json(file_path, orient="records", lines=True)
        return file_path

    def __len__(self):
        """Return the number of entries in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a dataset entry by index.
        Args:
            idx: The index of the entry to retrieve.
        Returns:
            The entry as a dictionary.
        """
        return self.data.iloc[idx].to_dict()


def get_evaluator_factories(judge_model):
    """Return a dictionary of evaluator factories for different evaluation types.
    Args:
        judge_model: The judge model configuration to use for evaluators that require it.
    Returns:
        A dictionary mapping evaluator names to factory functions.
    """
    return {
        # Performance and Quality (AI-assisted)
        "groundedness": lambda: GroundednessEvaluator(judge_model),
        "retrieval": lambda: RetrievalEvaluator(judge_model),
        "relevance": lambda: RelevanceEvaluator(judge_model),
        "coherence": lambda: CoherenceEvaluator(judge_model),
        "fluency": lambda: FluencyEvaluator(judge_model),
        # Performance and Quality (NLP)
        "f1_score": lambda: F1ScoreEvaluator(),
        "rouge_score": lambda: RougeScoreEvaluator(),
        "gleu_score": lambda: GleuScoreEvaluator(),
        "bleu_score": lambda: BleuScoreEvaluator(),
        "meteor_score": lambda: MeteorScoreEvaluator(),
        # Risk and Safety (AI-assisted)
        "violence": lambda: ViolenceEvaluator(judge_model),
        "sexual": lambda: SexualEvaluator(judge_model),
        "self_harm": lambda: SelfHarmEvaluator(judge_model),
        "hate_unfairness": lambda: HateUnfairnessEvaluator(judge_model),
        "indirect_attack": lambda: IndirectAttackEvaluator(judge_model),
        "protected_material": lambda: ProtectedMaterialEvaluator(judge_model),
        # Composite
        "qa_evaluator": lambda: QAEvaluator(judge_model),
        "content_safety": lambda: ContentSafetyEvaluator(judge_model),
        # Agent-specific
        "tool_call_accuracy": lambda: ToolCallAccuracyEvaluator(judge_model),
        "intent_resolution": lambda: IntentResolutionEvaluator(judge_model),
        "task_adherence": lambda: TaskAdherenceEvaluator(judge_model),
    }

class AiEvaluation(ABC):
    """Base class for all evaluation types."""
    def __init__(self, model_name: str = None, judge_model_name: str = "gpt-4.5-preview"):
        """Initialize the AiEvaluation base class.
        Args:
            model_name: The name of the model to evaluate.
            judge_model_name: The name of the judge model to use for evaluation.
        """
        self.model_name = model_name
        self.judge_model_name = judge_model_name
        self.evaluators = {}
        self.response_dataset = None
        self.azure_ai_project = AzureAIProject(
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
            project_name=os.getenv("AZURE_PROJECT_NAME")
        )

        # self.project_client = AIProjectClient.from_connection_string(
        #     credential=DefaultAzureCredential(),
        #     conn_str=os.environ["PROJECT_CONNECTION_STRING"],
        # )

    async def prepare_dataset(self, model_endpoints, queries, run_id=None):
        """Prepare a response dataset by generating responses for the given queries.
        Args:
            model_endpoints: The model endpoint(s) to use for generating responses.
            queries: A list of queries to generate responses for.
            run_id: Optional run identifier.
        Returns:
            The generated ResponseDataset.
        """
        dataset = ResponseDataset()
        await dataset.generate_responses(model_endpoints, queries)
        self.response_dataset = dataset
        return dataset

    def get_judge_model_configuration(self):
        """Get the configuration for the judge model.
        Returns:
            An AzureOpenAIModelConfiguration instance for the judge model.
        """
        return AzureOpenAIModelConfiguration(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=self.judge_model_name,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        )

    def add_evaluator(self, evaluator_name: str, evaluator_instance: Any):
        """Add an evaluator instance to the evaluation.
        Args:
            evaluator_name: The name of the evaluator.
            evaluator_instance: The evaluator instance to add.
        """
        self.evaluators[evaluator_name] = evaluator_instance

    def configure_evaluators(self, evaluator_config: Dict[str, bool]):
        """Configure evaluators based on the provided configuration.
        Args:
            evaluator_config: A dictionary mapping evaluator names to booleans indicating whether to use them.
        """
        judge_model = self.get_judge_model_configuration()
        factories = get_evaluator_factories(judge_model)
        for name, use_eval in evaluator_config.items():
            if use_eval and name in factories:
                self.add_evaluator(name, factories[name]())

    def process_evaluation_data(self, data):
        """Process input data into a format suitable for evaluation.
        Args:
            data: The input data to process. Can be a ResponseDataset, dict, DataFrame, list of dicts, or JSONL file path.
        Returns:
            A tuple of (eval_data DataFrame, temp_file_path to JSONL file).
        """
        if isinstance(data, ResponseDataset):
            self.response_dataset = data
            eval_data = data.data
        elif isinstance(data, dict):
            eval_data = pd.DataFrame([{k: data.get(k, "") for k in ["query", "context", "ground_truth", "response", "system_message"]}])
        elif isinstance(data, pd.DataFrame):
            eval_data = data
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            eval_data = pd.DataFrame(data)
        elif isinstance(data, str) and data.endswith(".jsonl"):
            eval_data = pd.read_json(data, lines=True)
        else:
            raise ValueError("Data must be a ResponseDataset, dictionary, pandas DataFrame, list of dictionaries, or a JSONL file path.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as temp_file:
            eval_data.to_json(temp_file.name, orient="records", lines=True)
            temp_file_path = temp_file.name
        return eval_data, temp_file_path

    def run_evaluation(self, data, run_id=None, evaluation_name=None):
        """Run the evaluation using the configured evaluators.
        Args:
            data: The data to evaluate.
            run_id: Optional run identifier.
            evaluation_name: Optional name for the evaluation run.
        Returns:
            The evaluation results.
        """
        eval_data, temp_file_path = self.process_evaluation_data(data)
        if run_id is None:
            run_id = random.randint(1111, 9999)
        if evaluation_name is None:
            evaluation_name = f"Eval-Run-{run_id}-{self.model_name}"
        evaluator_config = EvaluatorConfig(
            column_mapping={
                "response": "${data.response}",
                "context": "${data.context}",
                "query": "${data.query}",
                "ground_truth": "${data.ground_truth}"
            }
        )
        try:
            results = evaluate(
                evaluation_name=evaluation_name,
                data=temp_file_path,
                evaluators=self.evaluators,
                azure_ai_project=self.azure_ai_project,
                evaluator_config=evaluator_config,
            )
            if isinstance(results, dict):
                results["run_id"] = run_id
                results["model_name"] = self.model_name
        finally:
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
        return results

    @abstractmethod
    def evaluate(self, data: Dict) -> Dict:
        """Evaluate the target using the configured evaluators.
        Args:
            data: The data to evaluate.
        Returns:
            The evaluation results as a dictionary.
        """
        pass

    def get_available_evaluators(self) -> List[str]:
        """Get a list of available evaluator names.
        Returns:
            A list of evaluator names.
        """
        return list(self.evaluators.keys())

class ModelEvaluation(AiEvaluation):
    """Evaluation for language models."""
    def __init__(self, model_name: str, judge_model_name: str = "gpt-4.5-preview", response_dataset=None):
        """Initialize the ModelEvaluation class.
        Args:
            model_name: The name of the model to evaluate.
            judge_model_name: The name of the judge model to use.
            response_dataset: Optional pre-existing response dataset.
        """
        super().__init__(model_name, judge_model_name)
        self.response_dataset = response_dataset

    def evaluate(self, data, run_id=None):
        """Evaluate the model using the provided data.
        Args:
            data: The data to evaluate.
            run_id: Optional run identifier.
        Returns:
            The evaluation results.
        """
        evaluation_name = f"Eval-Run-{run_id or random.randint(1111, 9999)}-{self.model_name}"
        return self.run_evaluation(data, run_id, evaluation_name)

    def format_results(self, results: Dict) -> str:
        """Format the evaluation results into a human-readable summary.
        Args:
            results: The evaluation results dictionary.
        Returns:
            A formatted string summary of the results.
        """
        summary = f"# {self.model_name} Evaluation Results\n\n"
        model_response = ""
        if "rows" in results and len(results["rows"]) > 0:
            row = results["rows"][0]
            model_response = row.get("target_response", "Response not available")
        summary += f"## Model Response:\n{model_response}\n\n"
        summary += f"## Evaluation Metrics:\n\n---\n\n{results}"
        return summary

class PromptEvaluation(AiEvaluation):
    """Evaluation for prompt comparison."""
    def __init__(self, model_name: str, judge_model_name: str = "gpt-4.5-preview", response_dataset=None):
        """Initialize the PromptEvaluation class.
        Args:
            model_name: The name of the model to evaluate.
            judge_model_name: The name of the judge model to use.
            response_dataset: Optional pre-existing response dataset.
        """
        super().__init__(model_name, judge_model_name)
        self.response_dataset = response_dataset

    def evaluate(self, data, run_id=None):
        """Evaluate prompts by comparing responses for different system messages.
        Args:
            data: The data to evaluate.
            run_id: Optional run identifier.
        Returns:
            The evaluation results as a dictionary.
        """
        import pandas as pd
        if run_id is None:
            run_id = random.randint(1111, 9999)
        evaluation_results = {}
        if isinstance(data, ResponseDataset):
            self.response_dataset = data
            system_messages = data.data['system_message'].unique()
            for idx, system_message in enumerate(system_messages, start=1):
                key = f"system_message_{idx}"
                subset = data.data[data.data['system_message'] == system_message]
                results = self.run_evaluation(subset, run_id, f"PromptEval-Run-{run_id}-{self.model_name}-Message{idx}")
                evaluation_results[key] = results
                if isinstance(results, dict):
                    results["system_message"] = system_message
        else:
            for idx, item in enumerate(self._preprocess_input_data(data), start=1):
                key = f"system_message_{idx}"
                system_message = item.get("system_message", "")
                eval_data = pd.DataFrame([item])
                results = self.run_evaluation(eval_data, run_id, f"PromptEval-Run-{run_id}-{self.model_name}-Message{idx}")
                evaluation_results[key] = results
                if isinstance(results, dict):
                    results["system_message"] = system_message
        evaluation_results["run_id"] = run_id
        evaluation_results["model_name"] = self.model_name
        return evaluation_results

    def _preprocess_input_data(self, data):
        """Preprocess input data for prompt evaluation.
        Args:
            data: The input data to preprocess.
        Returns:
            A list of dictionaries with standardized fields for evaluation.
        """
        import pandas as pd
        if isinstance(data, dict):
            return [
                {
                    "query": data.get("prompt", "") or data.get("query", ""),
                    "system_message": data.get(key, ""),
                    "context": data.get("context", ""),
                    "ground_truth": data.get("ground_truth", ""),
                    "response": data.get(f"response_{idx}", "")
                }
                for idx, key in enumerate(["system_message_1", "system_message_2"], start=1)
                if key in data
            ]
        if isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        if isinstance(data, str) and data.endswith(".jsonl"):
            return pd.read_json(data, lines=True).to_dict('records')
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data
        return []

    def format_results(self, results: Dict) -> str:
        """Format the prompt evaluation results into a summary and table.
        Args:
            results: The evaluation results dictionary.
        Returns:
            A tuple of (DataFrame, formatted string summary).
        """
        import pandas as pd
        summary = f"# {self.model_name} Prompt Evaluation Results\n\n"
        data, dynamic_fields = [], set()
        for key in ["system_message_1", "system_message_2"]:
            if key in results:
                for row in results[key]["rows"]:
                    row_data = {
                        "System Message": row.get("inputs.system_message", "N/A"),
                        "Query": row.get("inputs.query", "N/A"),
                        "Response": row.get("outputs.response", "N/A"),
                    }
                    for k, v in row.items():
                        if k.startswith("outputs.") and k != "outputs.response":
                            field = k.split(".", 1)[1]
                            row_data[field] = v
                            dynamic_fields.add(field)
                    data.append(row_data)
        columns = ["System Message", "Query", "Response"] + list(dynamic_fields)
        df = pd.DataFrame(data, columns=columns)
        table = df.to_markdown(index=False)
        summary += table
        summary += "\n\n## Metrics and Studio URLs\n\n"
        for key in ["system_message_1", "system_message_2"]:
            if key in results:
                metrics = results[key].get("metrics", {})
                studio_url = results[key].get("studio_url", "N/A")
                summary += f"### {key.capitalize()}\n"
                for metric, value in metrics.items():
                    summary += f"- {metric}: {value}\n"
                summary += f"- AI Foundry URL: {studio_url}\n\n"
        summary += f"\n## Run ID: {results.get('run_id', 'N/A')}\n"
        return df, summary

class AgentEvaluation(AiEvaluation):
    """Evaluation for agents."""
    import json
    import os
    from jinja2 import Environment, FileSystemLoader
    from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread, AgentResponseItem
    from semantic_kernel.functions import kernel_function
    from semantic_kernel.contents.function_call_content import FunctionCallContent
    from semantic_kernel.contents.function_result_content import FunctionResultContent

    def __init__(self, judge_model_name: str = "gpt-4.1"):
        """Initialize the AgentEvaluation class.
        Args:
            model_name: The name of the agent model to evaluate.
            judge_model_name: The name of the judge model to use.
        """
        
        super().__init__(None, judge_model_name)

    def evaluate(self, agent: ChatCompletionAgent, thread: ChatHistoryAgentThread, evaluator_config) -> Dict:
        """Evaluate the agent using the configured evaluators.
        Args:
            agent: The agent to evaluate.
            evaluator_config: Configuration for the evaluators.
        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        self.configure_evaluators(evaluator_config)

        for evaluator in self.evaluators:
            # Implement evaluation logic for each evaluator here
            pass
        
        return self._evaluate_agent_thread(agent=agent, thread=thread)
    
    def _get_agent_functions(self, agent: ChatCompletionAgent) -> list:
        print(f"\n{80 * '='}")
        print("# Semantic Kernel was provided with the following tools:")
        
        functions = []
        for plugin in agent.kernel.plugins:
            functions_metadata = agent.kernel.plugins[plugin].get_functions_metadata()
            for function in functions_metadata:
                # Serialize metadata to a dictionary
                function_dict = function.model_dump()
                # function_dict["type"] = "tool_call"
                functions.append(function_dict)
        print(f"## tool_definitions : {functions}")

        return functions

    def _convert_sk_function_call_to_tool_call(self, function_call: FunctionCallContent, tool_definitions: list):
        """
        Converts a Semantic Kernel function call to a tool call format.
        """
        from jinja2 import Environment, FileSystemLoader
        import json
        # Set up Jinja2 environment to load templates from the templates directory
        templates_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
        env = Environment(loader=FileSystemLoader(templates_dir), trim_blocks=True, lstrip_blocks=True)

        # Render tool definitions template
        defs_template = env.get_template('tool_definitions.j2')
        defs_str = defs_template.render(tool_definitions=tool_definitions)
        # Parse JSON back to Python object
        defs_obj = json.loads(defs_str)

        # Prepare context for tool call template
        tool_call_ctx = {
            'tool_call_id': function_call["id"],
            # Extract only the function name substring after the plugin prefix (e.g., "MenuPlugin-...")
            'name': function_call["function"]["name"].split('-', 1)[-1],
            'arguments': function_call["function"]["arguments"]
    }
        call_template = env.get_template('tool_call.j2')
        call_str = call_template.render(tool_call=tool_call_ctx)
        call_obj = json.loads(call_str)

        # print(f"Converted tool_call: {call_obj}")
        # print(f"Converted tool_definitions: {defs_obj}")

        return {'tool_definitions': defs_obj, 'tool_call': call_obj}

    def _eval_tool_call_accuracy(self, query: str, response: str, tool_call: FunctionCallContent, tool_definitions: dict) -> str:
        """
        Evaluates the tool call and returns the result.
        """
        from azure.ai.evaluation import ToolCallAccuracyEvaluator
        model_config = self.get_judge_model_configuration()

        converted_tools = self._convert_sk_function_call_to_tool_call(tool_call, tool_definitions)
        converted_tool_definitions = converted_tools['tool_definitions']
        converted_tool_call = converted_tools['tool_call']

        print(80 * "=")
        print(f"# Starting Evaluation with converted tool call format:")
        print(f"\n## Tool call: {converted_tool_call}")
        print(f"\n## Query: {query}")
        print(f"\n## Response: {response}")
        print(f"\n## Tool Definitions: {converted_tool_definitions}")

        for evaluator_name, evaluator_instance in self.evaluators.items():
            print(f"## Evaluator: {evaluator_name}")
            if isinstance(evaluator_instance, ToolCallAccuracyEvaluator):
                eval_result = evaluator_instance(query=query, response=response, tool_definitions=converted_tool_definitions, tool_calls=converted_tool_call)
            elif isinstance(evaluator_instance, (IntentResolutionEvaluator, TaskAdherenceEvaluator)):
                eval_result = evaluator_instance(query=query, response=response, tool_definitions=converted_tool_definitions)
            else:
                # return error saying evaluator is not supported for this evaluation
                raise NotImplementedError(f"Evaluator {evaluator_name} is not supported for this evaluation.")
        
        print(80 * "-")
        print("# Evaluation result:")
        print(f"\n## Tool call accuracy: {eval_result}")
        print(80 * "-")
        return query[-1]["content"], response, eval_result
    
    def _evaluate_agent_thread(self, thread: ChatHistoryAgentThread, agent: ChatCompletionAgent) -> None:
        from semantic_kernel.contents import AuthorRole, TextContent, FunctionCallContent, FunctionResultContent

        agent_functions = self._get_agent_functions(agent)

        if thread:
            function_calls = []
            function_results = []
            user_input = []
            messages = thread._chat_history.messages
            # messages = thread.get_messages()
            # print(f"Messages: {messages}")
            print(f"Thread ID: {thread.id}")

            # Build conversation history for evaluation
            conversation_history = []

            eval_results = []
            eval = False
            for message in messages:
                # Build conversation history as we iterate
                if message.role == AuthorRole.USER:
                    # This is the user input
                    user_input.append(message.content)
                    print(f"User input: {user_input[-1]}")
                elif message.role == AuthorRole.ASSISTANT:
                    for item in message.items:
                        if isinstance(item, FunctionCallContent):
                            eval = True
                            function_call = item.to_dict()
                            function_calls.append(function_call)
                            print(f"\n{80 * '-'}")
                        elif isinstance(item, FunctionResultContent):
                            function_result = item.to_dict()
                            function_results.append(function_result)
                            print(f"Function result: {function_result.result}")
                            print(80 * "-")
                        elif isinstance(item, TextContent):
                            # Assuming it's a message from the agent
                            response = message.content
                            print(f"Response: {response}")
                            print(80 * "-")
                            if eval:
                                print("Evaluating the tool call...")
                                eval_results.append(self._eval_tool_call_accuracy(query=conversation_history, response=str(response), tool_call=function_call, tool_definitions=agent_functions))
                                eval = False
                conversation_history.append({
                    "role": str(message.role),
                    "content": message.content
                }) if message.content else None

        return eval_results

def create_evaluation(eval_type: str, model_name: str) -> AiEvaluation:
    """Factory to create the appropriate evaluation type.
    Args:
        eval_type: The type of evaluation ('model', 'prompt', or 'agent').
        model_name: The name of the model to evaluate.
    Returns:
        An instance of the appropriate AiEvaluation subclass.
    Raises:
        ValueError: If the evaluation type is unknown.
    """
    if eval_type == "model":
        return ModelEvaluation(model_name)
    if eval_type == "prompt":
        return PromptEvaluation(model_name)
    if eval_type == "agent":
        return AgentEvaluation(model_name)
    raise ValueError(f"Unknown evaluation type: {eval_type}")