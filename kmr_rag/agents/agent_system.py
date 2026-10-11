from typing import Any, Optional, Dict
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method
)
from llama_index.core.callbacks.base import global_stack_trace
import os
from llama_index.core.llms import ChatMessage
# from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core.prompts.base import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
from transformers import BitsAndBytesConfig
from pathlib import Path
from islab.nlp.cancer_registry.agents.toolspecs import CRCRetrievalEngineToolSpec, CancerRegistryLlamaDebugHandler
import json
import pickle
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.tools.utils import create_schema_from_function
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.objects import (
    ObjectIndex,
    SimpleToolNodeMapping,
    ObjectRetriever,
)
from llama_index.core.prompts import BasePromptTemplate

def messages_to_prompt(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in messages)
def completion_to_prompt(completion: str) -> str:
    return completion.strip()

# define a custom retriever with reranking
class CustomRetriever(BaseRetriever):
    def __init__(self, vector_retriever, postprocessor=None):
        self._vector_retriever = vector_retriever
        self._postprocessor = postprocessor #or CohereRerank(top_n=5)
        super().__init__()

    def _retrieve(self, query_bundle):
        retrieved_nodes = self._vector_retriever.retrieve(query_bundle)
        filtered_nodes = self._postprocessor.postprocess_nodes(
            retrieved_nodes, query_bundle=query_bundle
        )

        return filtered_nodes


# define a custom object retriever that adds in a query planning tool
class CustomObjectRetriever(ObjectRetriever):
    def __init__(self, retriever, object_node_mapping, all_tools, service_context=None):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping
        self._service_context = service_context

    def retrieve(self, query_bundle):
        nodes = self._retriever.retrieve(query_bundle)
        tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=tools, service_context=self._service_context
        )
        sub_question_description = f"""\
Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this \
tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
"""
        sub_question_tool = QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="compare_tool", description=sub_question_description
            ),
        )

        return tools + [sub_question_tool]

class CancerRegistryAgenetSystem:

    def __init__(self, doc_idx_llm = None, agent_llm = None, index_llm=None, embed_model="local:BAAI/bge-large-en-v1.5",
                 rerank_model = "BAAI/bge-reranker-large", logger= None):
        self.node_parser = CancerRegistryAgenetSystem.create_node_parser()
        if doc_idx_llm is None:
            doc_idx_llm = CancerRegistryAgenetSystem.create_doc_idx_llm(logger=logger)
        self.doc_idx_llm = doc_idx_llm
        # if logger:
        #     logger('產生報告代理人服務環境...', idx=0)
        #service_context = create_doc_idx_service_context(doc_idx_llm, node_parser)

        if logger:
            self.logger=logger
            self.llama_debug = CancerRegistryLlamaDebugHandler(print_trace_on_end=True, logger=logger)
            self.callback_manager = CallbackManager([self.llama_debug])
            self.service_context = ServiceContext.from_defaults(llm=doc_idx_llm, embed_model=embed_model, node_parser=self.node_parser,
                                                            callback_manager=self.callback_manager)
            self.node_parser.callback_manager = self.callback_manager
        else:
            self.service_context = ServiceContext.from_defaults(llm=doc_idx_llm, embed_model=embed_model, node_parser=self.node_parser)
        
        if agent_llm is None:
            agent_llm = CancerRegistryAgenetSystem.create_agent_llm(logger=logger)
        self.agent_llm = agent_llm

        if index_llm is None:
            index_llm = CancerRegistryAgenetSystem.create_index_agent_llm(logger=logger)
        self.index_llm = index_llm
        self.index_service_context = ServiceContext.from_defaults(llm=index_llm,
            embed_model=embed_model
        )
        self.rerank = SentenceTransformerRerank(top_n=6, model=rerank_model)

    @staticmethod
    def create_bitsandbytes_config():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
    @staticmethod
    def create_index_agent_llm(callback_manager = None, selected_model = "mistralai/Mistral-7B-Instruct-v0.1", logger = None):
        quantization_config = CancerRegistryAgenetSystem.create_bitsandbytes_config()
        if logger:
            logger('載入統整代理人模型...', idx =0)
        if callback_manager:
            return HuggingFaceLLM(
                context_window=4096,
                max_new_tokens=512,
                generate_kwargs={"temperature": 0.001, "do_sample": True},
                tokenizer_name=selected_model,
                model_name=selected_model,
                is_chat_model = True,
                #system_prompt=agent_system_prompt,
                #query_wrapper_prompt=query_wrapper_prompt,
                # Here is how to fix the alternative error show above.
                # We transform inputs into Llama2 format here
                messages_to_prompt = messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                device_map="auto", 
                callback_manager=callback_manager,
                model_kwargs={"quantization_config": quantization_config, "pad_token_id" : 2},
                )
        else:
            return HuggingFaceLLM(
                context_window=4096,
                max_new_tokens=512,
                generate_kwargs={"temperature": 0.001, "do_sample": True},
                tokenizer_name=selected_model,
                model_name=selected_model,
                is_chat_model = True,
                #system_prompt=agent_system_prompt,
                #query_wrapper_prompt=query_wrapper_prompt,
                # Here is how to fix the alternative error show above.
                # We transform inputs into Llama2 format here
                messages_to_prompt = messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                device_map="auto", 
                model_kwargs={"quantization_config": quantization_config, "pad_token_id" : 2},
                )


    @staticmethod
    def create_agent_llm(callback_manager = None, selected_model = "mistralai/Mistral-7B-Instruct-v0.1", logger = None):
        agent_system_prompt = '''As a cancer registrar agent, your primary responsibility is to accurately curate the cancer registry data in compliance with user-defined requirements and the document you manage. It's imperative that your data compilation strictly adheres to the options and categories specified by users. In situations where the correct code cannot be determined due to insufficient information, or the content is unrelated to the specific cancer type in question, your response should be 'NA'. All responses must be formatted exclusively as valid JSON objects. Outputs in any format other than JSON are not acceptable. Each response should include the key 'explan', containing a string that justifies your classification choice. Additionally, incorporate other keys as required by user specifications. Remember, the structure and content of your JSON response should reflect the precision and adherence to guidelines essential for this role.'''
        query_wrapper_prompt = PromptTemplate(
            "[INST]<<SYS>>\n" + agent_system_prompt + "<</SYS>>\n\n{query_str}[/INST]"
        )
        quantization_config = CancerRegistryAgenetSystem.create_bitsandbytes_config()
        if logger:
            logger('載入互動式代理人模型...', idx =0)
        if callback_manager:
            return HuggingFaceLLM(
                context_window=4096,
                max_new_tokens=512,    
                generate_kwargs={"temperature": 0.001, "do_sample": True},
                tokenizer_name=selected_model,
                model_name=selected_model,
                is_chat_model = True,
                callback_manager=callback_manager,
                system_prompt=agent_system_prompt,
                query_wrapper_prompt=query_wrapper_prompt,
                
                # Here is how to fix the alternative error show above.
                # We transform inputs into Llama2 format here
                messages_to_prompt = messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                device_map="auto", 
                model_kwargs={"quantization_config": quantization_config, "pad_token_id" : 2},
                )
        else:
            return HuggingFaceLLM(
                context_window=4096,
                max_new_tokens=512,    
                generate_kwargs={"temperature": 0.001, "do_sample": True},
                tokenizer_name=selected_model,
                model_name=selected_model,
                is_chat_model = True,

                system_prompt=agent_system_prompt,
                query_wrapper_prompt=query_wrapper_prompt,
                
                # Here is how to fix the alternative error show above.
                # We transform inputs into Llama2 format here
                messages_to_prompt = messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                device_map="auto", 
                model_kwargs={"quantization_config": quantization_config, "pad_token_id" : 2},
                )                        
        
    @staticmethod
    def create_doc_idx_llm(callback_manager = None, selected_model = "mistralai/Mistral-7B-Instruct-v0.1", logger = None):
        system_prompt = '''As a cancer registrar in this system, your primary responsibility is to accurately curate the cancer registry data in compliance with user-defined requirements. It's imperative that your data compilation strictly adheres to the options and categories specified by users. In situations where the correct code cannot be determined due to insufficient information, or the content is unrelated to the specific cancer type in question, your response should be 'NA'. All responses must be formatted exclusively as valid JSON objects. Outputs in any format other than JSON are not acceptable. Each response should include the key 'explan', containing a string that justifies your classification choice. Additionally, incorporate other keys as required by user specifications. Remember, the structure and content of your JSON response should reflect the precision and adherence to guidelines essential for this role.'''

        query_wrapper_prompt = PromptTemplate(
            "[INST]<<SYS>>\n" + system_prompt + "<</SYS>>\n\n{query_str}[/INST]"
        )

        quantization_config = CancerRegistryAgenetSystem.create_bitsandbytes_config()

        if logger:
            logger('載入報告代理人模型...', idx =0)
        if callback_manager:
            return HuggingFaceLLM(
                context_window=4096,
                max_new_tokens=512,    
                generate_kwargs={"temperature": 0.001, "do_sample": True},
                #query_wrapper_prompt=query_wrapper_prompt,
                tokenizer_name=selected_model,
                model_name=selected_model,
                device_map="auto", 
                callback_manager=callback_manager,
                system_prompt = system_prompt,   
                query_wrapper_prompt=query_wrapper_prompt,
                model_kwargs={"quantization_config": quantization_config, "pad_token_id" : 2},
            )
        else:
            return HuggingFaceLLM(
                context_window=4096,
                max_new_tokens=512,    
                generate_kwargs={"temperature": 0.001, "do_sample": True},
                #query_wrapper_prompt=query_wrapper_prompt,
                tokenizer_name=selected_model,
                model_name=selected_model,
                device_map="auto", 
                system_prompt = system_prompt,   
                query_wrapper_prompt=query_wrapper_prompt,
                model_kwargs={"quantization_config": quantization_config, "pad_token_id" : 2},
            )

    def build_agent_per_doc(self, doc, idx, nodes):
        file_path = Path(doc.metadata["file_path"])
        file_id = str(file_path.parent.stem) + "_" + str(file_path.stem)
        if self.logger:
            self.logger(f'產生報告 [{idx+1}] 的索引...')
        crc_tool_spec = CRCRetrievalEngineToolSpec(doc, idx, nodes, self.service_context, logger = self.logger)
        histology_analysis_file_path = f".agent/histology/{file_id}_{idx}"

        histology_template = """Your task is to provide the histology coding for patients diagnosed with colorectal cancer. Your response should be based on the nature of the report:
    - In cases where the report is not a pathology report or lacks sufficient information to ascertain the histology type, your response should be 'NA'. This indicates that histology coding cannot be determined from the provided document.
    - If the report is a pathology report that confirms a diagnosis of colorectal cancer, you are required to analyze the details within and respond with the most appropriate histology code."""
        # determine the histology
        if not os.path.exists(histology_analysis_file_path):
            if self.logger:
                self.logger(f'產生報告 {idx+1} 的摘要資訊...')
                self.logger(chat_msg = f'**使用下列提詞從報告 {idx+1} 檢索相關語句:** {histology_template}')
            Path(histology_analysis_file_path).parent.mkdir(parents=True, exist_ok=True)
            str_results = crc_tool_spec.get_histology_fn(histology_template)
            results = json.loads(str_results)
            if self.logger:
                self.logger(chat_msg = str_results, chat_role = 'assistant')
            pickle.dump(results, open(histology_analysis_file_path, "wb"))
        else:
            if self.logger:
                self.logger(f'載入已分析之報告 [{idx+1}] 摘要資訊...')
            results = pickle.load(open(histology_analysis_file_path, "rb"))

        if results['histology'] != "NaN":
            self.logger(chat_msg = f"**報告 {idx+1} 的摘要資訊:**\n{results['explan']}")
            crc_tool= crc_tool_spec.to_tool_list(
                func_to_metadata_mapping={'get_histology_fn': ToolMetadata(
                    name="get_histology",
                    # description=f"""\
                    #     Use this tool to look up biographical information about celebrities.
                    #     The vector database schema is given below:
                        
                    #     {vector_store_info.json()}
                        
                    #     {tool_spec.auto_retrieve_fn.__doc__}
                    # """,
                    description=(f"""Useful for answering questions related to the histology of colorectal cancer patients.
    The schema is given below:
    {crc_tool_spec.get_histology_fn.__doc__}
    """,
                                
                                ),# {doc.metadata['date']}"),
                    fn_schema=create_schema_from_function("get_histology", crc_tool_spec.get_histology_fn)
                )}
            )

            agent = CancerRegistryReActAgent.from_tools(crc_tool, llm=self.agent_llm, verbose=True,
                                        system_prompt=f"""You are a specialized agent designed to answer queries about the \
cancer registry coding results based on a patient's medical reports.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on any prior knowledge.\
        """)
            return agent, f"{file_id}_{idx}", results#, summary
        else:
            return None, None, None

    @staticmethod
    def create_node_parser(chunk_sizes=[2048, 1024, 512], callback_manager = None):
        return HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes, callback_manager= callback_manager)

    def create_doc_idx_service_context(self, llm, node_parser, embed_model="local:BAAI/bge-large-en-v1.5"):
        return ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)
    
    def set_logger(self, logger):
        self.logger = logger
        self.llama_debug = CancerRegistryLlamaDebugHandler(print_trace_on_end=True, logger=logger)
        self.service_context.callback_manager = CallbackManager([self.llama_debug])
        self.agent_llm.callback_manager = self.service_context.callback_manager
        self.doc_idx_llm.callback_manager = self.service_context.callback_manager
        self.index_llm.callback_manager = self.service_context.callback_manager
        self.index_service_context.callback_manager = self.service_context.callback_manager
        self.node_parser.callback_manager = self.service_context.callback_manager

    def build_agents(self, docs):
        agents_dict = {}
        extra_info_dict = {}

        for idx, doc in enumerate(docs):
            if self.logger:
                self.logger(f'建置報告 [{idx+1}] 的報告代理人...')
                self.logger(chat_msg= f'開始建立建置報告 [{idx+1}] 的報告代理人...', chat_role="system")
            nodes = self.node_parser.get_nodes_from_documents([doc])
            # all_nodes.extend(nodes)

            # ID will be base + parent
            agent, file_id, results = self.build_agent_per_doc(doc, idx, nodes)#, summary = build_agent_per_doc(doc, nodes, file_id)
            
            if agent is not None:
                agents_dict[file_id] = agent
                extra_info_dict[file_id] = {"summary": results, "nodes": nodes}
            # define tool for each document agent
        all_tools = []
        for file_base, agent in agents_dict.items():
            summary = extra_info_dict[file_base]["summary"]
            doc_tool = QueryEngineTool(
                        query_engine=agent,
                        metadata=ToolMetadata(
                            name=f"tool_{file_base}",
                            description=f'Provide a search tool for the document that can be summarized as follows: {summary["explan"]}',
                        ),
                    )
            all_tools.append(doc_tool)

        
        tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
        obj_index = ObjectIndex.from_objects(
        all_tools,
        tool_mapping,
        VectorStoreIndex,
        service_context = self.index_service_context
        )
        vector_node_retriever = obj_index.as_node_retriever(similarity_top_k=10)

        custom_node_retriever = CustomRetriever(vector_node_retriever, self.rerank)

    # wrap it with ObjectRetriever to return objects
        custom_obj_retriever = CustomObjectRetriever(
        custom_node_retriever, tool_mapping, all_tools, service_context = self.index_service_context
        )
        
        top_agent = CancerRegistryReActAgent.from_tools(
        tool_retriever=custom_obj_retriever,
        system_prompt=""" \
    You are an agent designed to answer queries about the documentation.
    Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

    """,
        llm=self.index_llm,
        verbose=True, logger=self.logger
    )

        return top_agent#agents_dict, extra_info_dict

##################################################################################################
"""ReAct agent worker."""

import asyncio
import uuid
from itertools import chain
from threading import Thread
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)

from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.llms import ChatResponse
from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import MessageRole
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.tools import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.core.tools.types import AsyncBaseTool
from llama_index.core.utils import print_text, unit_generator
from llama_index.core.agent.react.step import ReActAgentWorker


def add_user_step_to_reasoning(
    step: TaskStep,
    memory: BaseMemory,
    current_reasoning: List[BaseReasoningStep],
    verbose: bool = False,
) -> None:
    """Add user step to memory."""
    if "is_first" in step.step_state and step.step_state["is_first"]:
        # add to new memory
        memory.put(ChatMessage(content=step.input, role=MessageRole.USER))
        step.step_state["is_first"] = False
    else:
        reasoning_step = ObservationReasoningStep(observation=step.input)
        current_reasoning.append(reasoning_step)
        if verbose:
            print(f"Added user message to memory: {step.input}")


class CancerRegistryReActAgentWorker(ReActAgentWorker):
    """OpenAI Agent worker."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        logger = None
    ) -> None:
        self._llm = llm
        self.callback_manager = callback_manager or llm.callback_manager
        self._max_iterations = max_iterations
        self._react_chat_formatter = react_chat_formatter or ReActChatFormatter()
        self._output_parser = output_parser or ReActOutputParser()
        self._verbose = verbose

        if len(tools) > 0 and tool_retriever is not None:
            raise ValueError("Cannot specify both tools and tool_retriever")
        elif len(tools) > 0:
            self._get_tools = lambda _: tools
        elif tool_retriever is not None:
            tool_retriever_c = cast(ObjectRetriever[BaseTool], tool_retriever)
            self._get_tools = lambda message: tool_retriever_c.retrieve(message)
        else:
            self._get_tools = lambda _: []
        self.logger = logger

    @classmethod
    def from_tools(
        cls,
        tools: Optional[Sequence[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        logger = None,
        **kwargs: Any,
    ) -> "CancerRegistryReActAgentWorker":
        """Convenience constructor method from set of of BaseTools (Optional).

        NOTE: kwargs should have been exhausted by this point. In other words
        the various upstream components such as BaseSynthesizer (response synthesizer)
        or BaseRetriever should have picked up off their respective kwargs in their
        constructions.

        Returns:
            ReActAgent
        """
        llm = llm #or OpenAI(model=DEFAULT_MODEL_NAME)
        if callback_manager is not None:
            llm.callback_manager = callback_manager
        
        return cls(
            tools=tools or [],
            tool_retriever=tool_retriever,
            llm=llm,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose, logger=logger
        )

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        sources: List[ToolOutput] = []
        current_reasoning: List[BaseReasoningStep] = []
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

        # initialize task state
        task_state = {
            "sources": sources,
            "current_reasoning": current_reasoning,
            "new_memory": new_memory,
        }
        task.extra_state.update(task_state)

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            step_state={"is_first": True},
        )

    def get_tools(self, input: str) -> List[AsyncBaseTool]:
        """Get tools."""
        return [adapt_to_async_tool(t) for t in self._get_tools(input)]

    def _extract_reasoning_step(
        self, output: ChatResponse, is_streaming: bool = False
    ) -> Tuple[str, List[BaseReasoningStep], bool]:
        """
        Extracts the reasoning step from the given output.

        This method parses the message content from the output,
        extracts the reasoning step, and determines whether the processing is
        complete. It also performs validation checks on the output and
        handles possible errors.
        """
        if output.message.content is None:
            raise ValueError("Got empty message.")
        message_content = output.message.content
        current_reasoning = []
        try:
            reasoning_step = self._output_parser.parse(message_content, is_streaming)
        except BaseException as exc:
            raise ValueError(f"Could not parse output: {message_content}") from exc
        if self._verbose:
            print_text(f"{reasoning_step.get_content()}\n", color="pink")
        if self.logger:
            self.logger(chat_msg = f'**擷取分析結果:** {reasoning_step.get_content()}', chat_role = 'system')
            
        current_reasoning.append(reasoning_step)

        if reasoning_step.is_done:
            return message_content, current_reasoning, True

        reasoning_step = cast(ActionReasoningStep, reasoning_step)
        if not isinstance(reasoning_step, ActionReasoningStep):
            raise ValueError(f"Expected ActionReasoningStep, got {reasoning_step}")

        return message_content, current_reasoning, False

    def _process_actions(
        self,
        task: Task,
        tools: Sequence[AsyncBaseTool],
        output: ChatResponse,
        is_streaming: bool = False,
    ) -> Tuple[List[BaseReasoningStep], bool]:
        tools_dict: Dict[str, AsyncBaseTool] = {
            tool.metadata.get_name(): tool for tool in tools
        }
        _, current_reasoning, is_done = self._extract_reasoning_step(
            output, is_streaming
        )

        if is_done:
            return current_reasoning, True

        # call tool with input
        reasoning_step = cast(ActionReasoningStep, current_reasoning[-1])
        tool = tools_dict[reasoning_step.action]
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: reasoning_step.action_input,
                EventPayload.TOOL: tool.metadata,
            },
        ) as event:
            tool_output = tool.call(**reasoning_step.action_input)
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})

        task.extra_state["sources"].append(tool_output)

        observation_step = ObservationReasoningStep(observation=str(tool_output))
        current_reasoning.append(observation_step)
        if self._verbose:
            print_text(f"{observation_step.get_content()}\n", color="blue")
        return current_reasoning, False

    async def _aprocess_actions(
        self,
        task: Task,
        tools: Sequence[AsyncBaseTool],
        output: ChatResponse,
        is_streaming: bool = False,
    ) -> Tuple[List[BaseReasoningStep], bool]:
        tools_dict = {tool.metadata.name: tool for tool in tools}
        _, current_reasoning, is_done = self._extract_reasoning_step(
            output, is_streaming
        )

        if is_done:
            return current_reasoning, True

        # call tool with input
        reasoning_step = cast(ActionReasoningStep, current_reasoning[-1])
        tool = tools_dict[reasoning_step.action]
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: reasoning_step.action_input,
                EventPayload.TOOL: tool.metadata,
            },
        ) as event:
            tool_output = await tool.acall(**reasoning_step.action_input)
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})

        task.extra_state["sources"].append(tool_output)

        observation_step = ObservationReasoningStep(observation=str(tool_output))
        current_reasoning.append(observation_step)
        if self._verbose:
            print_text(f"{observation_step.get_content()}\n", color="blue")
        return current_reasoning, False

    def _get_response(
        self,
        current_reasoning: List[BaseReasoningStep],
        sources: List[ToolOutput],
    ) -> AgentChatResponse:
        """Get response from reasoning steps."""
        if len(current_reasoning) == 0:
            raise ValueError("No reasoning steps were taken.")
        elif len(current_reasoning) == self._max_iterations:
            raise ValueError("Reached max iterations.")

        if isinstance(current_reasoning[-1], ResponseReasoningStep):
            response_step = cast(ResponseReasoningStep, current_reasoning[-1])
            response_str = response_step.response
        else:
            response_str = current_reasoning[-1].get_content()

        # TODO: add sources from reasoning steps
        return AgentChatResponse(response=response_str, sources=sources)

    def _get_task_step_response(
        self, agent_response: AGENT_CHAT_RESPONSE_TYPE, step: TaskStep, is_done: bool
    ) -> TaskStepOutput:
        """Get task step response."""
        if is_done:
            new_steps = []
        else:
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    # NOTE: input is unused
                    input=None,
                )
            ]

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    def _infer_stream_chunk_is_final(self, chunk: ChatResponse) -> bool:
        """Infers if a chunk from a live stream is the start of the final
        reasoning step. (i.e., and should eventually become
        ResponseReasoningStep — not part of this function's logic tho.).

        Args:
            chunk (ChatResponse): the current chunk stream to check

        Returns:
            bool: Boolean on whether the chunk is the start of the final response
        """
        latest_content = chunk.message.content
        if latest_content:
            if not latest_content.startswith(
                "Thought"
            ):  # doesn't follow thought-action format
                return True
            else:
                if "Answer: " in latest_content:
                    return True
        return False

    def _add_back_chunk_to_stream(
        self, chunk: ChatResponse, chat_stream: Generator[ChatResponse, None, None]
    ) -> Generator[ChatResponse, None, None]:
        """Helper method for adding back initial chunk stream of final response
        back to the rest of the chat_stream.

        Args:
            chunk (ChatResponse): the chunk to add back to the beginning of the
                                    chat_stream.

        Return:
            Generator[ChatResponse, None, None]: the updated chat_stream
        """
        updated_stream = chain.from_iterable(  # need to add back partial response chunk
            [
                unit_generator(chunk),
                chat_stream,
            ]
        )
        # use cast to avoid mypy issue with chain and Generator
        updated_stream_c: Generator[ChatResponse, None, None] = cast(
            Generator[ChatResponse, None, None], updated_stream
        )
        return updated_stream_c

    async def _async_add_back_chunk_to_stream(
        self, chunk: ChatResponse, chat_stream: AsyncGenerator[ChatResponse, None]
    ) -> AsyncGenerator[ChatResponse, None]:
        """Helper method for adding back initial chunk stream of final response
        back to the rest of the chat_stream.

        NOTE: this itself is not an async function.

        Args:
            chunk (ChatResponse): the chunk to add back to the beginning of the
                                    chat_stream.

        Return:
            AsyncGenerator[ChatResponse, None]: the updated async chat_stream
        """
        yield chunk
        async for item in chat_stream:
            yield item

    def _run_step(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_reasoning(
                step,
                task.extra_state["new_memory"],
                task.extra_state["current_reasoning"],
                verbose=self._verbose,
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)
        if self.logger:
            #self.logger(mindmap=f'# {task.input}\n\n## 取得報告代理人列表\n\n')
            self.logger(chat_msg = f'{task.input}', chat_role = 'user')
            tool_list= ""
            for tool in tools:
                tool_list = tool_list + f'- {tool.metadata.name}: **{tool.metadata.description}**\n'
            #self.logger(mindmap=tool_list, append_mindmap = True)
            self.logger(chat_msg = f'**可用的工具列表如下:**\n{tool_list}', chat_role = 'system')
            
        input_chat = self._react_chat_formatter.format(
            tools,
            chat_history=task.memory.get() + task.extra_state["new_memory"].get_all(),
            current_reasoning=task.extra_state["current_reasoning"],
        )
        # send prompt
        chat_response = self._llm.chat(input_chat)
        if self.logger:
            self.logger(chat_msg=f'{messages_to_prompt(input_chat + [chat_response.message])}', chat_role = 'ai')
        
        # given react prompt outputs, call tools or return response
        reasoning_steps, is_done = self._process_actions(
            task, tools, output=chat_response
        )
        task.extra_state["current_reasoning"].extend(reasoning_steps)
        agent_response = self._get_response(
            task.extra_state["current_reasoning"], task.extra_state["sources"]
        )
        if self.logger:
            self.logger(chat_msg=f'**產生回覆:** {agent_response}', chat_role = 'system')
        if is_done:
            task.extra_state["new_memory"].put(
                ChatMessage(content=agent_response.response, role=MessageRole.ASSISTANT)
            )
            if self.logger:
                self.logger(chat_msg=f'**分析完成，加入記憶暫存:** {agent_response}', chat_role = 'system')

        return self._get_task_step_response(agent_response, step, is_done)

    async def _arun_step(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_reasoning(
                step,
                task.extra_state["new_memory"],
                task.extra_state["current_reasoning"],
                verbose=self._verbose,
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)

        input_chat = self._react_chat_formatter.format(
            tools,
            chat_history=task.memory.get() + task.extra_state["new_memory"].get_all(),
            current_reasoning=task.extra_state["current_reasoning"],
        )
        # send prompt
        chat_response = await self._llm.achat(input_chat)
        # given react prompt outputs, call tools or return response
        reasoning_steps, is_done = await self._aprocess_actions(
            task, tools, output=chat_response
        )
        task.extra_state["current_reasoning"].extend(reasoning_steps)
        agent_response = self._get_response(
            task.extra_state["current_reasoning"], task.extra_state["sources"]
        )
        if is_done:
            task.extra_state["new_memory"].put(
                ChatMessage(content=agent_response.response, role=MessageRole.ASSISTANT)
            )

        return self._get_task_step_response(agent_response, step, is_done)

    def _run_step_stream(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_reasoning(
                step,
                task.extra_state["new_memory"],
                task.extra_state["current_reasoning"],
                verbose=self._verbose,
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)

        input_chat = self._react_chat_formatter.format(
            tools,
            chat_history=task.memory.get() + task.extra_state["new_memory"].get_all(),
            current_reasoning=task.extra_state["current_reasoning"],
        )

        chat_stream = self._llm.stream_chat(input_chat)

        # iterate over stream, break out if is final answer after the "Answer: "
        full_response = ChatResponse(
            message=ChatMessage(content=None, role="assistant")
        )
        is_done = False
        for latest_chunk in chat_stream:
            full_response = latest_chunk
            is_done = self._infer_stream_chunk_is_final(latest_chunk)
            if is_done:
                break

        if not is_done:
            # given react prompt outputs, call tools or return response
            reasoning_steps, _ = self._process_actions(
                task, tools=tools, output=full_response, is_streaming=True
            )
            task.extra_state["current_reasoning"].extend(reasoning_steps)
            # use _get_response to return intermediate response
            agent_response: AGENT_CHAT_RESPONSE_TYPE = self._get_response(
                task.extra_state["current_reasoning"], task.extra_state["sources"]
            )
        else:
            # Get the response in a separate thread so we can yield the response
            response_stream = self._add_back_chunk_to_stream(
                chunk=latest_chunk, chat_stream=chat_stream
            )

            agent_response = StreamingAgentChatResponse(
                chat_stream=response_stream,
                sources=task.extra_state["sources"],
            )
            thread = Thread(
                target=agent_response.write_response_to_history,
                args=(task.extra_state["new_memory"],),
            )
            thread.start()

        return self._get_task_step_response(agent_response, step, is_done)

    async def _arun_step_stream(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_reasoning(
                step,
                task.extra_state["new_memory"],
                task.extra_state["current_reasoning"],
                verbose=self._verbose,
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)

        input_chat = self._react_chat_formatter.format(
            tools,
            chat_history=task.memory.get() + task.extra_state["new_memory"].get_all(),
            current_reasoning=task.extra_state["current_reasoning"],
        )

        chat_stream = await self._llm.astream_chat(input_chat)

        # iterate over stream, break out if is final answer after the "Answer: "
        full_response = ChatResponse(
            message=ChatMessage(content=None, role="assistant")
        )
        is_done = False
        async for latest_chunk in chat_stream:
            full_response = latest_chunk
            is_done = self._infer_stream_chunk_is_final(latest_chunk)
            if is_done:
                break

        if not is_done:
            # given react prompt outputs, call tools or return response
            reasoning_steps, _ = self._process_actions(
                task, tools=tools, output=full_response, is_streaming=True
            )
            task.extra_state["current_reasoning"].extend(reasoning_steps)
            # use _get_response to return intermediate response
            agent_response: AGENT_CHAT_RESPONSE_TYPE = self._get_response(
                task.extra_state["current_reasoning"], task.extra_state["sources"]
            )
        else:
            # Get the response in a separate thread so we can yield the response
            response_stream = self._async_add_back_chunk_to_stream(
                chunk=latest_chunk, chat_stream=chat_stream
            )

            agent_response = StreamingAgentChatResponse(
                achat_stream=response_stream,
                sources=task.extra_state["sources"],
            )
            # create task to write chat response to history
            asyncio.create_task(
                agent_response.awrite_response_to_history(
                    task.extra_state["new_memory"]
                )
            )
            # wait until response writing is done
            await agent_response._is_function_false_event.wait()

        return self._get_task_step_response(agent_response, step, is_done)

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        return self._run_step(step, task)

    @trace_method("run_step")
    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        return await self._arun_step(step, task)

    @trace_method("run_step")
    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        # TODO: figure out if we need a different type for TaskStepOutput
        return self._run_step_stream(step, task)

    @trace_method("run_step")
    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        return await self._arun_step_stream(step, task)

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""
        # add new messages to memory
        task.memory.set(task.memory.get() + task.extra_state["new_memory"].get_all())
        # reset new memory
        task.extra_state["new_memory"].reset()
        
import uuid
from abc import abstractmethod
from collections import ChainMap, deque
from typing import Any, Deque, Dict, List, Optional, Union, cast

from llama_index.core.agent.types import (
    BaseAgent,
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    ChatResponseMode,
    StreamingAgentChatResponse,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory


class BaseAgentRunner(BaseAgent):
    """Base agent runner."""

    @abstractmethod
    def create_task(self, input: str, **kwargs: Any) -> Task:
        """Create task."""

    @abstractmethod
    def delete_task(
        self,
        task_id: str,
    ) -> None:
        """Delete task.

        NOTE: this will not delete any previous executions from memory.

        """

    @abstractmethod
    def list_tasks(self, **kwargs: Any) -> List[Task]:
        """List tasks."""

    @abstractmethod
    def get_task(self, task_id: str, **kwargs: Any) -> Task:
        """Get task."""

    @abstractmethod
    def get_upcoming_steps(self, task_id: str, **kwargs: Any) -> List[TaskStep]:
        """Get upcoming steps."""

    @abstractmethod
    def get_completed_steps(self, task_id: str, **kwargs: Any) -> List[TaskStepOutput]:
        """Get completed steps."""

    def get_completed_step(
        self, task_id: str, step_id: str, **kwargs: Any
    ) -> TaskStepOutput:
        """Get completed step."""
        # call get_completed_steps, and then find the right task
        completed_steps = self.get_completed_steps(task_id, **kwargs)
        for step_output in completed_steps:
            if step_output.task_step.step_id == step_id:
                return step_output
        raise ValueError(f"Could not find step_id: {step_id}")

    @abstractmethod
    def run_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step."""

    @abstractmethod
    async def arun_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (async)."""

    @abstractmethod
    def stream_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (stream)."""

    @abstractmethod
    async def astream_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (async stream)."""

    @abstractmethod
    def finalize_response(
        self,
        task_id: str,
        step_output: Optional[TaskStepOutput] = None,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Finalize response."""

    @abstractmethod
    def undo_step(self, task_id: str) -> None:
        """Undo previous step."""
        raise NotImplementedError("undo_step not implemented")


def validate_step_from_args(
    task_id: str, input: Optional[str] = None, step: Optional[Any] = None, **kwargs: Any
) -> Optional[TaskStep]:
    """Validate step from args."""
    if step is not None:
        if input is not None:
            raise ValueError("Cannot specify both `step` and `input`")
        if not isinstance(step, TaskStep):
            raise ValueError(f"step must be TaskStep: {step}")
        return step
    else:
        return (
            None
            if input is None
            else TaskStep(
                task_id=task_id, step_id=str(uuid.uuid4()), input=input, **kwargs
            )
        )


class TaskState(BaseModel):
    """Task state."""

    task: Task = Field(..., description="Task.")
    step_queue: Deque[TaskStep] = Field(
        default_factory=deque, description="Task step queue."
    )
    completed_steps: List[TaskStepOutput] = Field(
        default_factory=list, description="Completed step outputs."
    )


class AgentState(BaseModel):
    """Agent state."""

    task_dict: Dict[str, TaskState] = Field(
        default_factory=dict, description="Task dictionary."
    )

    def get_task(self, task_id: str) -> Task:
        """Get task state."""
        return self.task_dict[task_id].task

    def get_completed_steps(self, task_id: str) -> List[TaskStepOutput]:
        """Get completed steps."""
        return self.task_dict[task_id].completed_steps

    def get_step_queue(self, task_id: str) -> Deque[TaskStep]:
        """Get step queue."""
        return self.task_dict[task_id].step_queue


class AgentRunner(BaseAgentRunner):
    """Agent runner.

    Top-level agent orchestrator that can create tasks, run each step in a task,
    or run a task e2e. Stores state and keeps track of tasks.

    Args:
        agent_worker (BaseAgentWorker): step executor
        chat_history (Optional[List[ChatMessage]], optional): chat history. Defaults to None.
        state (Optional[AgentState], optional): agent state. Defaults to None.
        memory (Optional[BaseMemory], optional): memory. Defaults to None.
        llm (Optional[LLM], optional): LLM. Defaults to None.
        callback_manager (Optional[CallbackManager], optional): callback manager. Defaults to None.
        init_task_state_kwargs (Optional[dict], optional): init task state kwargs. Defaults to None.

    """

    # # TODO: implement this in Pydantic

    def __init__(
        self,
        agent_worker: BaseAgentWorker,
        chat_history: Optional[List[ChatMessage]] = None,
        state: Optional[AgentState] = None,
        memory: Optional[BaseMemory] = None,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        init_task_state_kwargs: Optional[dict] = None,
        delete_task_on_finish: bool = False,
    ) -> None:
        """Initialize."""
        self.agent_worker = agent_worker
        self.state = state or AgentState()
        self.memory = memory or ChatMemoryBuffer.from_defaults(chat_history, llm=llm)
        self.callback_manager = callback_manager or CancerRegistryCallbackManager([])
        self.init_task_state_kwargs = init_task_state_kwargs or {}
        self.delete_task_on_finish = delete_task_on_finish

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self.memory.get_all()

    def reset(self) -> None:
        self.memory.reset()

    def create_task(self, input: str, **kwargs: Any) -> Task:
        """Create task."""
        task = Task(
            input=input,
            memory=self.memory,
            extra_state=self.init_task_state_kwargs,
            **kwargs,
        )
        # # put input into memory
        # self.memory.put(ChatMessage(content=input, role=MessageRole.USER))

        # get initial step from task, and put it in the step queue
        initial_step = self.agent_worker.initialize_step(task)
        task_state = TaskState(
            task=task,
            step_queue=deque([initial_step]),
        )
        # add it to state
        self.state.task_dict[task.task_id] = task_state

        return task

    def delete_task(
        self,
        task_id: str,
    ) -> None:
        """Delete task.

        NOTE: this will not delete any previous executions from memory.

        """
        self.state.task_dict.pop(task_id)

    def list_tasks(self, **kwargs: Any) -> List[Task]:
        """List tasks."""
        return list(self.state.task_dict.values())

    def get_task(self, task_id: str, **kwargs: Any) -> Task:
        """Get task."""
        return self.state.get_task(task_id)

    def get_upcoming_steps(self, task_id: str, **kwargs: Any) -> List[TaskStep]:
        """Get upcoming steps."""
        return list(self.state.get_step_queue(task_id))

    def get_completed_steps(self, task_id: str, **kwargs: Any) -> List[TaskStepOutput]:
        """Get completed steps."""
        return self.state.get_completed_steps(task_id)

    def _run_step(
        self,
        task_id: str,
        step: Optional[TaskStep] = None,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Execute step."""
        task = self.state.get_task(task_id)
        step_queue = self.state.get_step_queue(task_id)
        step = step or step_queue.popleft()

        # TODO: figure out if you can dynamically swap in different step executors
        # not clear when you would do that by theoretically possible

        if mode == ChatResponseMode.WAIT:
            cur_step_output = self.agent_worker.run_step(step, task, **kwargs)
        elif mode == ChatResponseMode.STREAM:
            cur_step_output = self.agent_worker.stream_step(step, task, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # append cur_step_output next steps to queue
        next_steps = cur_step_output.next_steps
        step_queue.extend(next_steps)

        # add cur_step_output to completed steps
        completed_steps = self.state.get_completed_steps(task_id)
        completed_steps.append(cur_step_output)

        return cur_step_output

    async def _arun_step(
        self,
        task_id: str,
        step: Optional[TaskStep] = None,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Execute step."""
        task = self.state.get_task(task_id)
        step_queue = self.state.get_step_queue(task_id)
        step = step or step_queue.popleft()

        # TODO: figure out if you can dynamically swap in different step executors
        # not clear when you would do that by theoretically possible
        if mode == ChatResponseMode.WAIT:
            cur_step_output = await self.agent_worker.arun_step(step, task, **kwargs)
        elif mode == ChatResponseMode.STREAM:
            cur_step_output = await self.agent_worker.astream_step(step, task, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # append cur_step_output next steps to queue
        next_steps = cur_step_output.next_steps
        step_queue.extend(next_steps)

        # add cur_step_output to completed steps
        completed_steps = self.state.get_completed_steps(task_id)
        completed_steps.append(cur_step_output)

        return cur_step_output

    def run_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step."""
        step = validate_step_from_args(task_id, input, step, **kwargs)
        return self._run_step(task_id, step, mode=ChatResponseMode.WAIT, **kwargs)

    async def arun_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (async)."""
        step = validate_step_from_args(task_id, input, step, **kwargs)
        return await self._arun_step(
            task_id, step, mode=ChatResponseMode.WAIT, **kwargs
        )

    def stream_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (stream)."""
        step = validate_step_from_args(task_id, input, step, **kwargs)
        return self._run_step(task_id, step, mode=ChatResponseMode.STREAM, **kwargs)

    async def astream_step(
        self,
        task_id: str,
        input: Optional[str] = None,
        step: Optional[TaskStep] = None,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        step = validate_step_from_args(task_id, input, step, **kwargs)
        return await self._arun_step(
            task_id, step, mode=ChatResponseMode.STREAM, **kwargs
        )

    def finalize_response(
        self,
        task_id: str,
        step_output: Optional[TaskStepOutput] = None,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Finalize response."""
        if step_output is None:
            step_output = self.state.get_completed_steps(task_id)[-1]
        if not step_output.is_last:
            raise ValueError(
                "finalize_response can only be called on the last step output"
            )

        if not isinstance(
            step_output.output,
            (AgentChatResponse, StreamingAgentChatResponse),
        ):
            raise ValueError(
                "When `is_last` is True, cur_step_output.output must be "
                f"AGENT_CHAT_RESPONSE_TYPE: {step_output.output}"
            )

        # finalize task
        self.agent_worker.finalize_task(self.state.get_task(task_id))

        if self.delete_task_on_finish:
            self.delete_task(task_id)

        return cast(AGENT_CHAT_RESPONSE_TYPE, step_output.output)

    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Chat with step executor."""
        if chat_history is not None:
            self.memory.set(chat_history)
        task = self.create_task(message)

        result_output = None
        while True:
            # pass step queue in as argument, assume step executor is stateless
            cur_step_output = self._run_step(task.task_id, mode=mode)

            if cur_step_output.is_last:
                result_output = cur_step_output
                break

        return self.finalize_response(task.task_id, result_output)

    async def _achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Chat with step executor."""
        if chat_history is not None:
            self.memory.set(chat_history)
        task = self.create_task(message)

        result_output = None
        while True:
            # pass step queue in as argument, assume step executor is stateless
            cur_step_output = await self._arun_step(task.task_id, mode=mode)

            if cur_step_output.is_last:
                result_output = cur_step_output
                break

        return self.finalize_response(task.task_id, result_output)

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message, chat_history, tool_choice, mode=ChatResponseMode.WAIT
            )
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = await self._achat(
                message, chat_history, tool_choice, mode=ChatResponseMode.WAIT
            )
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> StreamingAgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message, chat_history, tool_choice, mode=ChatResponseMode.STREAM
            )
            assert isinstance(chat_response, StreamingAgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> StreamingAgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = await self._achat(
                message, chat_history, tool_choice, mode=ChatResponseMode.STREAM
            )
            assert isinstance(chat_response, StreamingAgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    def undo_step(self, task_id: str) -> None:
        """Undo previous step."""
        raise NotImplementedError("undo_step not implemented")

"""ReAct agent.

Simple wrapper around AgentRunner + ReActAgentWorker.

For the legacy implementation see:
```python
from llama_index.agent.legacy.react.base import ReActAgent
```

"""
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Type,
)

from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser
#from llama_index.agent.runner.base import AgentRunner
from llama_index.core.callbacks import (
    CallbackManager,
)
from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.tools import BaseTool

class CancerRegistryReActAgent(ReActAgent):
    """ReAct agent.

    Subclasses AgentRunner with a ReActAgentWorker.

    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        memory: BaseMemory,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        logger = None
    ) -> None:
        """Init params."""
        callback_manager = callback_manager or llm.callback_manager
        step_engine = CancerRegistryReActAgentWorker.from_tools(
            tools=tools,
            tool_retriever=tool_retriever,
            llm=llm,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose, logger=logger
        )
        AgentRunner.__init__(
            self, step_engine,
            memory=memory,
            llm=llm,
            callback_manager=callback_manager,
        )
    
    @classmethod
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False, logger = None,
        **kwargs: Any,
    ) -> "CancerRegistryReActAgent":
        """Convenience constructor method from set of of BaseTools (Optional).

        NOTE: kwargs should have been exhausted by this point. In other words
        the various upstream components such as BaseSynthesizer (response synthesizer)
        or BaseRetriever should have picked up off their respective kwargs in their
        constructions.

        Returns:
            ReActAgent
        """
        llm = llm# or OpenAI(model=DEFAULT_MODEL_NAME)
        if callback_manager is not None:
            llm.callback_manager = callback_manager
        memory = memory or memory_cls.from_defaults(
            chat_history=chat_history or [], llm=llm
        )
        return cls(
            tools=tools or [],
            tool_retriever=tool_retriever,
            llm=llm,
            memory=memory,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose, logger=logger
        )
