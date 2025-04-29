from dotenv import load_dotenv
from utils.util import load_dotenv_from_azd
import streamlit as st

import pandas as pd
import random
from utils.evaluation import ModelEvaluation, PromptEvaluation, AgentEvaluation
from utils.models import SUPPORTED_MODELS
import logging
import asyncio

# Load environment variables from .env file
load_dotenv_from_azd()

# Function to get evaluator configuration from UI selections
def get_evaluator_config():
    return {
        # Performance and Quality (AI-assisted)
        "groundedness": st.session_state.get("groundedness", True),
        "retrieval": st.session_state.get("retrieval", False),
        "relevance": st.session_state.get("relevance", False),
        "coherence": st.session_state.get("coherence", False),
        "fluency": st.session_state.get("fluency", False),
        
        # Performance and Quality (NLP)
        "f1_score": st.session_state.get("f1_score", False),
        "rouge_score": st.session_state.get("rouge_score", False),
        "gleu_score": st.session_state.get("gleu_score", False),
        "bleu_score": st.session_state.get("bleu_score", False),
        "meteor_score": st.session_state.get("meteor_score", False),
        
        # Risk and Safety (AI-assisted)
        "violence": st.session_state.get("violence", False),
        "sexual": st.session_state.get("sexual", False),
        "self_harm": st.session_state.get("self_harm", False),
        "hate_unfairness": st.session_state.get("hate_unfairness", False),
        "indirect_attack": st.session_state.get("indirect_attack", False),
        "protected_material": st.session_state.get("protected_material", False),
        
        # Composite
        "qa_evaluator": st.session_state.get("qa_evaluator", False),
        "content_safety": st.session_state.get("content_safety", False),

        # Agents
        "tool_call_accuracy": st.session_state.get("tool_call_accuracy", False),
        "intent_resolution": st.session_state.get("intent_resolution", False),
        "task_adherence": st.session_state.get("task_adherence", False),
    }

# Function to prepare datasets and run evaluations
async def prepare_and_evaluate(model_name, judge_llm, queries, run_id=None):
    """
    Prepare a dataset by generating responses and run evaluation.
    
    Args:
        model_name: Name of the model to evaluate
        judge_llm: Name of the judge model to use
        queries: List of query dictionaries
        run_id: Optional run ID for grouping evaluations
        
    Returns:
        Tuple of (dataset, evaluation_results, formatted_results)
    """
    from utils.models import ModelEndpoints
    
    # Create evaluation and endpoints
    model_eval = ModelEvaluation(model_name, judge_llm)
    model_eval.configure_evaluators(get_evaluator_config())
    model_endpoints = ModelEndpoints(model_name)
    
    # Generate responses
    dataset = await model_eval.prepare_dataset(model_endpoints, queries)
    
    # Use shared run_id if provided
    if run_id is None:
        run_id = random.randint(1111, 9999)
        
    # Run evaluation
    results = model_eval.evaluate(dataset, run_id=run_id)
    
    # Format results
    try:
        formatted_results = pd.DataFrame(results["rows"])
        formatted_results = formatted_results.loc[:, formatted_results.columns.str.startswith("outputs")]
    except:
        formatted_results = pd.DataFrame({"Error": ["Failed to extract results"]})
    
    return dataset, results, formatted_results

# Helper function to run async code in Streamlit
def run_async(coro):
    """Run an async function in a synchronous context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# Function to format and display evaluation results
def format_and_display_results(results, show_transposed=True):
    """
    Format and display evaluation results in a consistent way.
    
    Args:
        results: Dictionary containing evaluation results with model/system message keys
        show_transposed: Whether to show the dataframe transposed
    """
    for key, result in results.items():
        st.write(f"### {key}")
        
        # Display evaluation metrics
        if "data" in result and not result["data"].empty:
            if show_transposed:
                # Transpose for better readability
                transposed_df = result["data"].T
                st.dataframe(transposed_df)
            else:
                st.dataframe(result["data"])
        else:
            st.write("No evaluation metrics available")
        
        # Display Azure AI Foundry URL if available
        if result.get("studio_url"):
            st.write("**Azure AI Foundry:**")
            st.markdown(f"[Open in Azure AI Foundry]({result['studio_url']})")
        
        st.markdown("---")

# Set up the main menu
menu = st.sidebar.selectbox("Menu", ["Development", "Production"])

# -------------------- Menu: Development --------------------
if menu == "Development":
    st.title("Development")
    # Create tabs for the Development section
    model_tab, prompt_tab, agent_tab = st.tabs(["Evaluate Models", "Evaluate Prompts", "Evaluate Agents"])
    
    # Add evaluator selection to the sidebar
    st.sidebar.subheader("Select Evaluators")

    # Group evaluators by category
    # -------------------- Sidebar: Performance and Quality --------------------
    with st.sidebar.expander("Performance and Quality (AI-assisted)", expanded=True):
        st.checkbox("Groundedness", value=True, key="groundedness")
        st.checkbox("Retrieval", key="retrieval")
        st.checkbox("Relevance", key="relevance")
        st.checkbox("Coherence", key="coherence")
        st.checkbox("Fluency", key="fluency")

    with st.sidebar.expander("Performance and Quality (NLP)", expanded=False):
        st.checkbox("F1Score", key="f1_score")
        st.checkbox("RougeScore", key="rouge_score")
        st.checkbox("GleuScore", key="gleu_score")
        st.checkbox("BleuScore", key="bleu_score")
        st.checkbox("MeteorScore", key="meteor_score")

    # -------------------- Sidebar: Risk and Safety --------------------
    with st.sidebar.expander("Risk and Safety (AI-assisted)", expanded=False):
        st.checkbox("Violence", key="violence")
        st.checkbox("Sexual", key="sexual")
        st.checkbox("Self Harm", key="self_harm")
        st.checkbox("Hate & Unfairness", key="hate_unfairness")
        st.checkbox("Indirect Attack", key="indirect_attack")
        st.checkbox("Protected Material", key="protected_material")

    # -------------------- Sidebar: Agents --------------------
    with st.sidebar.expander("Agents", expanded=False):
        st.checkbox("Tool Call Accuracy", key="tool_call_accuracy")
        st.checkbox("Intent Resolution", key="intent_resolution")
        st.checkbox("Task Adherence", key="task_adherence")

    # -------------------- Sidebar: Composite --------------------
    with st.sidebar.expander("Composite", expanded=False):
        st.checkbox("Q&A", key="qa_evaluator")
        st.checkbox("Content Safety", key="content_safety")

    # -------------------- Tab: Model Evaluation --------------------
    with model_tab:
        st.header("Compare & Evaluate Models")
        
        # Create two columns for the dropdowns
        col1, col2 = st.columns(2)
        
        with col1:
            model1 = st.selectbox(
                "Select Model 1",
                SUPPORTED_MODELS,
                key="model1"
            )
            
        with col2:
            model2 = st.selectbox(
                "Select Model 2",
                SUPPORTED_MODELS,
                index=1 if len(SUPPORTED_MODELS) > 1 else 0,
                key="model2"
            )
        
        # Add a dropdown to select the judge LLM under "Evaluate Models"
        judge_llm = st.selectbox(
            "Select Judge LLM",
            options=SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index("gpt-4.1"),
            help="Choose the LLM to be used as the judge for evaluation."
        )
        if judge_llm in ["o4-mini", "o3"]:
            st.warning("Reasoning models (o4-mini, o3) are currently not supported as judge LLMs.")
        
        # Add some space before data input options
        st.write("")
        
        # Add option to choose between uploading a dataset or manual entry
        st.subheader("Test Data")
        data_input_method = st.radio(
            "Choose how to provide test data:",
            ["Upload Test Dataset", "Enter Manually"],
            horizontal=True
        )
        
        if data_input_method == "Upload Test Dataset":
            uploaded_file = st.file_uploader("Upload a .jsonl file", type="jsonl", key="model_upload")
            
            if uploaded_file is not None:
                import pandas as pd
                from utils.models import ModelEndpoints
                from utils.evaluation import ResponseDataset
                
                # Read queries
                queries_data = pd.read_json(uploaded_file, lines=True)
                st.write("### Queries Preview")
                st.dataframe(queries_data.head())
                
                # Convert DataFrame to list of dictionaries
                queries = queries_data.to_dict(orient="records")
                
                # Add a Generate and Evaluate button
                if st.button("Generate Responses and Evaluate", key="generate_evaluate"):
                    with st.spinner(f"Generating responses and evaluating {model1} and {model2}..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Use shared run_id for both evaluations
                        run_id = random.randint(1111, 9999)
                        
                        # Evaluate both models using common run_id
                        evaluation_results = {}
                        
                        # Process model1
                        status_text.text(f"Processing {model1}...")
                        model1_dataset, model1_results, model1_df = run_async(
                            prepare_and_evaluate(model1, judge_llm, queries, run_id)
                        )
                        progress_bar.progress(50)
                        
                        # Process model2
                        status_text.text(f"Processing {model2}...")
                        model2_dataset, model2_results, model2_df = run_async(
                            prepare_and_evaluate(model2, judge_llm, queries, run_id)
                        )
                        progress_bar.progress(90)
                        
                        # Display response previews
                        st.write(f"### {model1} Responses Preview")
                        st.dataframe(model1_dataset.data.head())
                        
                        st.write(f"### {model2} Responses Preview")
                        st.dataframe(model2_dataset.data.head())
                        
                        # Store results in session state
                        st.session_state.evaluation_results = {
                            model1: {
                                "data": model1_df,
                                "studio_url": model1_results.get("studio_url", "")
                            },
                            model2: {
                                "data": model2_df,
                                "studio_url": model2_results.get("studio_url", "")
                            }
                        }
                        
                        progress_bar.progress(100)
                        status_text.text("Evaluation completed!")
                        
                        # Display evaluation results using the common formatter
                        st.subheader("Evaluation Results")
                        format_and_display_results(st.session_state.evaluation_results)
                        
                        # Option to download generated datasets
                        col1, col2 = st.columns(2)
                        with col1:
                            model1_jsonl = model1_dataset.to_jsonl()
                            with open(model1_jsonl, "r") as f:
                                st.download_button(
                                    label=f"Download {model1} Responses",
                                    data=f,
                                    file_name=f"{model1}_responses.jsonl",
                                    mime="application/json"
                                )
                        with col2:
                            model2_jsonl = model2_dataset.to_jsonl()
                            with open(model2_jsonl, "r") as f:
                                st.download_button(
                                    label=f"Download {model2} Responses",
                                    data=f,
                                    file_name=f"{model2}_responses.jsonl",
                                    mime="application/json"
                                )
        
        else:  # Enter Manually
            st.write("Enter test data manually:")
            
            # Query input
            query = st.text_input(
                "Query",
                placeholder="Enter the user query here..."
            )
            
            # Context input
            context = st.text_area(
                "Context",
                placeholder="Enter any additional context information here...",
                height=150
            )
            
            # Submit button for manual entry
            if st.button("Evaluate", key="manual_evaluate"):
                # Check if query is provided
                if not query:
                    st.error("Please enter a query before evaluating.")
                else:
                    with st.spinner(f"Evaluating {model1} and {model2}..."):
                        # Set up test data
                        eval_data = [{
                            "query": query,
                            "context": context,
                            "ground_truth": ""  # We don't have ground truth for manual entry
                        }]

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Use shared run_id for both evaluations
                        run_id = random.randint(1111, 9999)
                        
                        # Process model1
                        status_text.text(f"Processing {model1}...")
                        model1_dataset, model1_results, model1_df = run_async(
                            prepare_and_evaluate(model1, judge_llm, eval_data, run_id)
                        )
                        progress_bar.progress(40)
                        
                        # Process model2
                        status_text.text(f"Processing {model2}...")
                        model2_dataset, model2_results, model2_df = run_async(
                            prepare_and_evaluate(model2, judge_llm, eval_data, run_id)
                        )
                        progress_bar.progress(80)
                        
                        # Display responses immediately after generation
                        model1_responses = model1_dataset.data['response'].tolist() if 'response' in model1_dataset.data.columns else ["No response available"]
                        model2_responses = model2_dataset.data['response'].tolist() if 'response' in model2_dataset.data.columns else ["No response available"]
                        
                        # Create a container for immediate response display
                        immediate_responses = st.container()
                        with immediate_responses:
                            st.subheader("Model Responses (Generated)")
                            st.info("Evaluation metrics will appear below once processing is complete.")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"### {model1}")
                                for i, resp in enumerate(model1_responses):
                                    st.text_area(f"{model1} Response", resp, height=150, key=f"immediate_{model1}_response_{i}", disabled=True)
                            
                            with col2:
                                st.write(f"### {model2}")
                                for i, resp in enumerate(model2_responses):
                                    st.text_area(f"{model2} Response", resp, height=150, key=f"immediate_{model2}_response_{i}", disabled=True)
                        
                        # Store in session state
                        st.session_state.evaluation_results = {
                            model1: {
                                "data": model1_df,
                                "studio_url": model1_results.get("studio_url", ""),
                                "responses": model1_responses
                            },
                            model2: {
                                "data": model2_df,
                                "studio_url": model2_results.get("studio_url", ""),
                                "responses": model2_responses
                            }
                        }
                        
                        progress_bar.progress(100)
                        status_text.text("Evaluation completed!")
                        
                        # Display the results using common formatter function
                        format_and_display_results(st.session_state.evaluation_results)
    
    # -------------------- Tab: Prompt Evaluation --------------------
    with prompt_tab:
        st.header("Compare & Evaluate Prompts")

        # Select Model
        selected_model = st.selectbox(
            "Select Model",
            SUPPORTED_MODELS,
            help="Choose the model to evaluate prompts."
        )

        # Add a dropdown to select the judge LLM for prompt evaluation
        prompt_judge_llm = st.selectbox(
            "Select Judge LLM",
            options=SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index("gpt-4.1"),
            key="prompt_judge_llm",
            help="Choose the LLM to be used as the judge for prompt evaluation."
        )
        if prompt_judge_llm in ["o4-mini", "o3"]:
            st.warning("Reasoning models (o4-mini, o3) are currently not supported as judge LLMs.")
        
        # System Message inputs
        sys_msg_col1, sys_msg_col2 = st.columns(2)
        with sys_msg_col1:
            system_message_1 = st.text_area(
                "System Message 1",
                placeholder="Enter the first system message...",
                height=100
            )
        with sys_msg_col2:
            system_message_2 = st.text_area(
                "System Message 2",
                placeholder="Enter the second system message...",
                height=100
            )

        # Prompt input
        prompt_input = st.text_input(
            "Prompt",
            placeholder="Enter the prompt to evaluate..."
        )

        # Context input (optional)
        context_input = st.text_area(
            "Context (Optional)",
            placeholder="Enter any additional context information here...",
            height=100
        )

        # Evaluate button
        if st.button("Evaluate", key="prompt_evaluate"):
            if not prompt_input:
                st.error("Please enter a prompt before evaluating.")
            elif not system_message_1 and not system_message_2:
                st.error("Please enter at least one system message before evaluating.")
            else:
                with st.spinner(f"Evaluating prompts with {selected_model}..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create query objects for each system message
                    queries_with_system_messages = []
                    
                    if system_message_1:
                        queries_with_system_messages.append({
                            "query": prompt_input,
                            "system_message": system_message_1,
                            "context": context_input,
                            "ground_truth": ""  # Optional ground truth
                        })
                    
                    if system_message_2:
                        queries_with_system_messages.append({
                            "query": prompt_input,
                            "system_message": system_message_2,
                            "context": context_input,
                            "ground_truth": ""  # Optional ground truth
                        })
                    
                    # Create the prompt evaluation and run ID
                    run_id = random.randint(1111, 9999)
                    prompt_eval = PromptEvaluation(selected_model, prompt_judge_llm)
                    prompt_eval.configure_evaluators(get_evaluator_config())
                    
                    # Create model endpoint
                    status_text.text("Creating model endpoint...")
                    from utils.models import ModelEndpoints
                    model_endpoints = ModelEndpoints(selected_model)
                    progress_bar.progress(10)
                    
                    # Generate responses for each system message
                    status_text.text("Generating responses for each system message...")
                    dataset = run_async(prompt_eval.prepare_dataset(
                        model_endpoints, 
                        queries_with_system_messages, 
                        run_id=run_id
                    ))
                    progress_bar.progress(50)
                    
                    # Display generated responses immediately
                    response_display = st.container()
                    with response_display:
                        st.subheader("Generated Responses")
                        st.info("Evaluation metrics will appear below once processing is complete.")
                        
                        response_cols = st.columns(len(queries_with_system_messages))
                        
                        for i, (col, query) in enumerate(zip(response_cols, queries_with_system_messages)):
                            # Filter dataset to get response for this system message
                            system_message = query["system_message"]
                            matching_rows = dataset.data[dataset.data['system_message'] == system_message]
                            
                            if not matching_rows.empty:
                                response = matching_rows.iloc[0]['response']
                                with col:
                                    st.write(f"### System Message {i+1}")
                                    st.text_area(
                                        f"Response for System Message {i+1}", 
                                        response, 
                                        height=150, 
                                        key=f"response_{i+1}", 
                                        disabled=True
                                    )
                    
                    # Run the evaluation using the generated dataset
                    status_text.text("Evaluating responses...")
                    results = prompt_eval.evaluate(dataset, run_id=run_id)
                    progress_bar.progress(90)
                    
                    # Format results in a consistent way with model evaluation
                    system_message_results = {}
                    
                    # Extract data for each system message
                    for idx, system_msg_key in enumerate(["system_message_1", "system_message_2"]):
                        if system_msg_key in results:
                            try:
                                result_df = pd.DataFrame(results[system_msg_key]["rows"])
                                result_df = result_df.loc[:, result_df.columns.str.startswith("outputs")]
                                
                                # Use consistent naming for display
                                display_key = f"System Message {idx+1}"
                                system_message_results[display_key] = {
                                    "data": result_df,
                                    "studio_url": results[system_msg_key].get("studio_url", "")
                                }
                            except:
                                display_key = f"System Message {idx+1}"
                                system_message_results[display_key] = {
                                    "data": pd.DataFrame({"Error": ["Failed to extract results"]}),
                                    "studio_url": ""
                                }
                    
                    # Store in session state
                    st.session_state.prompt_evaluation_results = system_message_results
                    
                    progress_bar.progress(100)
                    status_text.text("Evaluation completed!")
                    
                    # Display evaluation results using common formatter
                    st.subheader("Evaluation Results")
                    format_and_display_results(st.session_state.prompt_evaluation_results)
                    
                    # Option to download generated dataset
                    dataset_jsonl = dataset.to_jsonl()
                    with open(dataset_jsonl, "r") as f:
                        st.download_button(
                            label="Download Responses as JSONL",
                            data=f,
                            file_name=f"prompt_eval_responses_{run_id}.jsonl",
                            mime="application/json"
                        )
                    
                    status_text.text("Evaluation completed!")
    
    # -------------------- Tab: Agent Evaluation --------------------
    with agent_tab:
        st.header("Evaluate Agents")
        # Select Judge LLM for Agent Evaluation
        agent_judge_llm = st.selectbox(
            "Select Judge LLM for Agent Evaluation",
            SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index("gpt-4.1"),
            key="agent_judge_llm",
            help="Choose the judge LLM to evaluate agent tool calls."
        )
        if agent_judge_llm in ["o4-mini", "o3"]:
            st.warning("Reasoning models (o4-mini, o3) are currently not supported as judge LLMs.")
        
        # Add a button to trigger the chat completion agent
        if st.button("Run Chat Completion Agent", key="run_chat_agent"):
            with st.spinner("Running Chat Completion Agent..."):
                from evalagents.chat_completion_agent import run_chat_agent, USER_INPUTS

                # Run the agent to get agent instance, thread, and responses
                agent, thread, responses = run_async(
                    run_chat_agent(model_name="gpt-4.1", user_inputs=USER_INPUTS)
                )

                # Display conversation
                st.subheader("Agent Conversation")
                for user_input, agent_name, response in responses:
                    st.write(f"**User:** {user_input}")
                    st.write(f"**{agent_name}:** {response}")
                st.success("✅ Conversation completed!")

            # Evaluate agent interactions
            with st.spinner("Evaluating Agent..."):
                evaluator_config = get_evaluator_config()
                agent_eval = AgentEvaluation(agent_judge_llm)
                eval_results = agent_eval.evaluate(agent, thread, evaluator_config)

                # Prepare display format
                import pandas as pd
                formatted = {}
                st.subheader("Agent Evaluation Results")
                for result in eval_results:
                    st.json(result)
                st.success("✅ Evaluations completed!")



# -------------------- Menu: Production --------------------
elif menu == "Production":
    st.title("Production")
    st.write("Production environment settings and options will be available here.")

logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
