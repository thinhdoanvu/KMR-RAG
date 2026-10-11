from pathlib import Path
from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import StorageContext, load_index_from_storage
import os
from llama_index.core.prompts.base import PromptTemplate
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, cast
from llama_index.core.schema import NodeWithScore
from llama_index.core.callbacks import LlamaDebugHandler
from llama_index.core.callbacks.schema import (
    BASE_TRACE_EVENT,
    TIMESTAMP_FORMAT,
    CBEvent,
    CBEventType,
    EventStats,
    EventPayload,
)

def update_query_engine_prompts(tool_prompt, query_engine):
    qa_prompt_tmpl_str = """Context information is below.
---------------------
{context_str}
---------------------
Given the context information and the following knowledge and command, answer the query.
Knowledge and command: {cancer_registry_kg}
Query: {query_str}
Answer: \
"""
    prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    partial_prompt_tmpl = prompt_tmpl.partial_format(cancer_registry_kg=tool_prompt)
    #print(partial_prompt_tmpl)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": partial_prompt_tmpl})

    refine_prompt_tmpl_str = """The original query is as follows: {query_str}
We have provided an existing answer: {existing_answer}
Knowledge and command: {cancer_registry_kg}
We have the opportunity to refine the existing answer (only if needed) with some more context below.
------------
{context_msg}
------------
Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.
Refined Answer:\
"""
    prompt_tmpl = PromptTemplate(refine_prompt_tmpl_str)
    partial_prompt_tmpl = prompt_tmpl.partial_format(cancer_registry_kg=tool_prompt)
    #print(partial_prompt_tmpl)
    query_engine.update_prompts({"response_synthesizer:refine_template": partial_prompt_tmpl})
    	

class CancerRegistryLlamaDebugHandler(LlamaDebugHandler):
    """Callback handler that keeps track of debug info.

    This handler simply keeps track of event starts/ends, separated by event types.
    You can use this callback handler to keep track of and debug events.

    Args:
        event_starts_to_ignore (Optional[List[CBEventType]]): list of event types to
            ignore when tracking event starts.
        event_ends_to_ignore (Optional[List[CBEventType]]): list of event types to
            ignore when tracking event ends.

    """

    def __init__(
        self,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
        print_trace_on_end: bool = True,
        logger = None
    ) -> None:
        """Initialize the llama debug handler."""
        self._logger = logger
        self._event_pairs_by_type: Dict[CBEventType, List[CBEvent]] = defaultdict(list)
        self._event_pairs_by_id: Dict[str, List[CBEvent]] = defaultdict(list)
        self._sequential_events: List[CBEvent] = []
        self._cur_trace_id: Optional[str] = None
        self._trace_map: Dict[str, List[str]] = defaultdict(list)
        self.print_trace_on_end = print_trace_on_end
        event_starts_to_ignore = (
            event_starts_to_ignore if event_starts_to_ignore else []
        )
        event_ends_to_ignore = event_ends_to_ignore if event_ends_to_ignore else []
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore,
            event_ends_to_ignore=event_ends_to_ignore,
        )

    def get_events(self, event_type: Optional[CBEventType] = None) -> List[CBEvent]:
        """Get all events for a specific event type."""
        if event_type is not None:
            return self._event_pairs_by_type[event_type]

        return self._sequential_events

    def _get_time_stats_from_event_pairs(
        self, event_pairs: List[List[CBEvent]]
    ) -> EventStats:
        """Calculate time-based stats for a set of event pairs."""
        total_secs = 0.0
        for event_pair in event_pairs:
            start_time = datetime.strptime(event_pair[0].time, TIMESTAMP_FORMAT)
            end_time = datetime.strptime(event_pair[-1].time, TIMESTAMP_FORMAT)
            total_secs += (end_time - start_time).total_seconds()

        return EventStats(
            total_secs=total_secs,
            average_secs=total_secs / len(event_pairs),
            total_count=len(event_pairs),
        )

    def get_llm_inputs_outputs(self) -> List[List[CBEvent]]:
        """Get the exact LLM inputs and outputs."""
        return self._get_event_pairs(self._event_pairs_by_type[CBEventType.LLM])

    def get_event_time_info(
        self, event_type: Optional[CBEventType] = None
    ) -> EventStats:
        event_pairs = self.get_event_pairs(event_type)
        return self._get_time_stats_from_event_pairs(event_pairs)

    def flush_event_logs(self) -> None:
        """Clear all events from memory."""
        self._event_pairs_by_type = defaultdict(list)
        self._event_pairs_by_id = defaultdict(list)
        self._sequential_events = []

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Launch a trace."""
        self._trace_map = defaultdict(list)
        self._cur_trace_id = trace_id

    def _get_time_stats_from_event_pairs(
        self, event_pairs: List[List[CBEvent]]
    ) -> EventStats:
        """Calculate time-based stats for a set of event pairs."""
        total_secs = 0.0
        for event_pair in event_pairs:
            start_time = datetime.strptime(event_pair[0].time, TIMESTAMP_FORMAT)
            end_time = datetime.strptime(event_pair[-1].time, TIMESTAMP_FORMAT)
            total_secs += (end_time - start_time).total_seconds()

        return EventStats(
            total_secs=total_secs,
            average_secs=total_secs / len(event_pairs),
            total_count=len(event_pairs),
        )
    
    def _print_trace_map(self, cur_event_id: str, level: int = 0) -> None:
        """Recursively print trace map to terminal for debugging."""
        event_pair = self._event_pairs_by_id[cur_event_id]
        if event_pair:
            time_stats = self._get_time_stats_from_event_pairs([event_pair])
            indent = " " * level * 2
            print(
                f"{indent}|_{event_pair[0].event_type} -> ",
                f"{time_stats.total_secs} seconds",
                flush=True,
            )

        child_event_ids = self._trace_map[cur_event_id]
        for child_event_id in child_event_ids:
            self._print_trace_map(child_event_id, level=level + 1)

    def get_event_pairs(
        self, event_type: Optional[CBEventType] = None
    ) -> List[List[CBEvent]]:
        """Pair events by ID, either all events or a specific type."""
        if event_type is not None:
            return self._get_event_pairs(self._event_pairs_by_type[event_type])

        return self._get_event_pairs(self._sequential_events)

    def add_indent_to_text(self, text, level: int = 0):
        span = "\t" * level
        if isinstance(text, str):
            text = text.replace('\n', '\n'+span)
            return f"{span}{text}"
        elif isinstance(text, List):
            return [span + t.replace('\n', '\n'+span) for t in text]
        
    def _generate_trace_map_string(self, cur_event_id: str, level: int = 0) -> None:
        """Recursively print trace map to terminal for debugging."""
        event_pair = self._event_pairs_by_id[cur_event_id]
        output_strs = []
        if event_pair:
            try:
                time_stats = self._get_time_stats_from_event_pairs([event_pair])
                #indent = ">" * level
                #output_str = f"{indent} {event_pair[0].event_type} ({time_stats.total_secs} s):\n"
                output_str = f"{level}. {event_pair[0].event_type} ({time_stats.total_secs} s):\n"
                if event_pair[0].event_type == CBEventType.RETRIEVE:
                    output_str = f'{output_str}\t- **使用如下查詢字串來取得參考資料**:\n\t\t```\n{self.add_indent_to_text(event_pair[0].payload.get(EventPayload.QUERY_STR), 2)}\n\t\t```\n'
                    output_str = f'{output_str}\t- **取得如下參考資料**:\n'
                    nodes = cast(List[NodeWithScore], event_pair[1].payload.get(EventPayload.NODES, []))
                    for idx, node in enumerate(nodes):                    
                        output_str = f'{output_str}\t\t{idx+1}. **文字內容**:\n\t\t\t```\n{self.add_indent_to_text(node.node.text, 3)}\n\t\t\t```\n'
                        output_str = f'{output_str}\t\t\t- **來源**: {node.metadata["file_path"]}\n'
                elif event_pair[0].event_type == CBEventType.AGENT_STEP:
                    output_str = f'{output_str}\t- 處理訊息：{self.add_indent_to_text(event_pair[0].payload.get(EventPayload.MESSAGES), 1)}\n'
                    if event_pair[1].event_type == CBEventType.AGENT_STEP:
                        output_str = f'{output_str}\t- 回覆：{self.add_indent_to_text(event_pair[1].payload.get(EventPayload.RESPONSE), 1)}\n'
                elif event_pair[0].event_type == CBEventType.LLM:
                    output_str = f'{output_str}\t- **代理人分析中**\n\t\t```\n{self.add_indent_to_text(event_pair[0].payload.get(EventPayload.PROMPT), 2)}\n\t\t```\n'
                    output_str = f'{output_str}\t- **分析結果**:\n\t\t```\n{self.add_indent_to_text(event_pair[1].payload.get(EventPayload.COMPLETION).text, 2)}\n\t\t```\n'
                elif event_pair[0].event_type == CBEventType.RERANKING:
                    output_str = f'{output_str}\t- 重新排序取得的資料 (top_k = {event_pair[0].payload.get(EventPayload.TOP_K, "")})\n'
                    output_str = f'{output_str}\t- **重新排序後的參考資料如下**:\n'
                    nodes = cast(List[NodeWithScore], event_pair[1].payload.get(EventPayload.NODES, []))
                    for idx, node in enumerate(nodes):                    
                        output_str = f'{output_str}\t\t{idx+1}. **文字內容**:\n\t\t\t```\n{self.add_indent_to_text(node.node.text, 3)}\n\t\t\t```\n\n'
                        output_str = f'{output_str}\t\t\t- **來源:** {node.metadata["file_path"]}\n'
                elif event_pair[0].event_type == CBEventType.SYNTHESIZE:
                    output_str = f'{output_str}\t- **合成取得的資料**:\n'
                    output_str = f'{output_str}\t- **結果**:\n\t\t```{self.add_indent_to_text(event_pair[1].payload.get(EventPayload.RESPONSE).response, 2)}\n\t\t```\n'
                elif event_pair[0].event_type == CBEventType.CHUNKING:
                    pass
                elif event_pair[0].event_type == CBEventType.QUERY:
                    pass
                output_strs.append(output_str)
            except:
                return output_str
        child_event_ids = self._trace_map[cur_event_id]
        for child_event_id in child_event_ids:
            output_strs.extend(self._generate_trace_map_string(child_event_id, level=level + 1))
        
        return output_strs


    def print_trace_map(self) -> None:
        """Print simple trace map to terminal for debugging of the most recent trace."""
        print("*" * 10, flush=True)
        print(f"Trace: {self._cur_trace_id}", flush=True)
        output_strs = self._generate_trace_map_string(BASE_TRACE_EVENT, level=1)
        self._logger(chat_msg = '\n'.join(output_strs), chat_role = 'assistant')

        # for idx, output in enumerate(output_strs):
        #     if idx:
        #         self._logger(chat_msg = output, chat_role = 'assistant')
        #     else:
        #         self._logger(chat_msg = f'**AI 回覆:** {output}', chat_role = 'assistant')
        

class CRCRetrievalEngineToolSpec(BaseToolSpec):
    """Tool spec."""

    spec_functions = ["get_histology_fn"]
    ccr_histology_template ='''Your primary task is to accurately curate the histology code for patients diagnosed with colorectal cancer. This involves meticulously reviewing and analyzing the given medical record of the patient to determine the most appropriate histology code. Your decision should be based on a thorough assessment of clinical findings and any relevant medical descriptions that shed light on the histological nature of the colorectal cancer. It is essential to use only the information provided in the document at hand.
When responding, you are required to select the most appropriate option from the following standard codes, which represents different histology types of colorectal cancer:

- 8000: Use this code if the histology type is "Neoplasm, malignant", "Blastoma, NOS", "Tumor, malignant, NOS", "Cancer", "Malignancy", "Unclassified tumor, malignant".
- 8010: Use this code if the histology type is "Carcinoma in situ, NOS", "Intraepithelial carcinoma, NOS", "Carcinoma, NOS", "Epithelial tumor, malignant", or "Heterotopia-associated carcinoma".
- 8013: Use this code if the histology type is "Large cell neuroendocrine carcinoma", or "Combined large cell neuroendocrine carcinoma".
- 8014: Use this code if the histology type is "Large cell carcinoma with rhabdoid phenotype".
- 8020: Use this code if the histology type is "Carcinoma, undifferentiated, NOS", "Carcinoma, poorly differentiated, NOS", "Anaplastic undifferentiated carcinoma", or "Dedifferentiated carcinoma".
- 8070: Use this code if the histology type is "squamous cell carcinoma in situ, NOS", "epidermoid carcinoma in situ, NOS", "intraepidermal carcinoma, NOS", "intraepithelial squamous cell carcinoma", "squamous cell carcinoma, NOS", "squamous cell carcinoma, usual type", "epidermoid carcinoma, NOS", "squamous carcinoma", or "squamous cell epithelioma".
- 8120: Use this code if the histology type is "Transitional cell carcinoma in situ", "Urothelial carcinoma in situ", "Transitional cell carcinoma, NOS", "Urothelial carcinoma, NOS", or "Transitional carcinoma".
- 8140: Use this code if the histology type is "Adenocarcinoma in situ, NOS", "Adenocarcinoma, NOS", "Adenocarcinoma, usual type", "Carcinoma of Skene, Cowper and Littre Glands", "Endolymphatic sac tumor", "Parathyroid carcinoma", or "Acinar adenocarcinoma of prostate".
- 8210: Use this code if the histology type is "Adenocarcinoma in situ in adenomatous polyp", "Adenocarcinoma in situ in a polyp, NOS", "Carcinoma in situ in a polyp, NOS", "Adenocarcinoma in situ in polypoid adenoma", "Adenocarcinoma in situ in tubular adenoma", "Carcinoma in situ in adenomatous polyp", "Adenocarcinoma in adenomatous polyp", "Adenocarcinoma in a polyp, NOS", "Carcinoma in a polyp, NOS", or "Adenocarcinoma in polypoid adenoma".
- 8211: Use this code if the histology type is "Tubular adenocarcinoma", or "Tubular carcinoma".
- 8244: Use this code if the histology type is "Mixed adenoneuroendocrine carcinoma", "Combined carcinoid and adenocarcinoma", "Combined/mixed carcinoid and adenocarcinoma", "Composite carcinoid", "MANEC", or "Mixed carcinoid-adenocarcinoma"
- 8246: Use this code if the histology type is "Neuroendocrine carcinoma, NOS", or "Poorly differentiated neuroendocrine neoplasm".
- 8261: Use this code if the histology type is "Adenocarcinoma in situ in villous adenoma", or "Adenocarcinoma in villous adenoma".
- 8263: Use this code if the histology type is "Adenocarcinoma in situ in tubulovillous adenoma", "Adenocarcinoma in tubulovillous adenoma", "Villoglandular carcinoma", "Villoglandular variant of endometrioid adenocarcinoma", "Papillotubular adenocarcinoma", or "Tubulopapillary adenocarcinoma".
- 8480: Use this code if the histology type is "Mucinous adenocarcinoma", "Acinar adenocarcinoma, mucinous variant", "Colloid adenocarcinoma", "Colloid carcinoma", "Gelatinous adenocarcinoma", "Gelatinous carcinoma", "Mucinous carcinoma", "Mucoid adenocarcinoma", "Mucoid carcinoma", "Mucous adenocarcinoma", "Mucous carcinoma", "Pseudomyxoma peritonei with unknown primary site", "Mucinous tubular and spindle cell carcinoma".
- 8490: Use this code if the histology type is "Signet ring cell carcinoma", "Signet ring cell adenocarcinoma", "Acinar adenocarcinoma, signet ring-like variant", "Mucinous carcinoma, signet ring cell type", "Poorly cohesive carcinoma", or "Signet ring cell/histiocytoid carcinoma".
- 8510: Use this code if the histology type is "Medullary carcinoma, NOS", "Medullary adenocarcinoma", or "Medullary-like carcinoma".

Your response should be limited to one of these codes, aligning with the most accurate and specific histological classification of the cancer case. For example, if the case is pathologically diagnosed as 'Adenocarcinoma in situ in adenomatous polyp', your response should be in the following JSON format:

{
"explan": "[Your explanation here based on the document]",
"cite": "[Cited passages from the document used for your decision]",
"histology": "8210"
}

Alternatively, if the case is diagnosed as 'Adenocarcinoma in situ' without further specification, your response should be:

{
"explan": "[Your explanation here based on the document]",
"cite": "[Cited passages from the document used for your decision]",
"histology": "8140"
}'''

    def __init__(
        self,
        #index: BaseIndex,  # TODO typing
        doc,
        idx,
        nodes,
        auto_merging_context,
        similarity_top_k = 6,
        rerank = SentenceTransformerRerank(top_n=6, model="BAAI/bge-reranker-large"),
        logger=None
    ) -> None:
        """Initialize with parameters."""
        file_path = Path(doc.metadata["file_path"])
        file_id = str(file_path.parent.stem) + "_" + str(file_path.stem)        
        vi_out_path = f".data/indexes/{file_id}_{idx}"
        
        self.logger=logger
        
        if not os.path.exists(vi_out_path):
            leaf_nodes = get_leaf_nodes(nodes)        
            storage_context = StorageContext.from_defaults() # �N index �x�s�b�O���餤
            storage_context.docstore.add_documents(nodes)
            # https://docs.llamaindex.ai/en/stable/api_reference/indices/vector_store.html#llama_index.indices.vector_store.base.VectorStoreIndex
            # �o��ڭ̴��� leaf_nodes �@�� embedding index �� node�Astorage context ���h�O�Ҧ��� nodes �]�A��ӵ��c�A�̫� service_context �O��� service �Ψ쪺�����귽
            self._index = VectorStoreIndex(
                    leaf_nodes,
                    service_context=auto_merging_context,
                    storage_context=storage_context
                )
            self._index.storage_context.persist(persist_dir=vi_out_path)
        else:
            self._index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=vi_out_path),
                service_context=auto_merging_context,
            )
        self._context = auto_merging_context
        self.similarity_top_k = similarity_top_k
        self.rerank = rerank
        

    def get_histology_fn(
        self,
        query: str,
#        top_k: int,
#        filter_key_list: List[str],
#        filter_value_list: List[str],
    ) -> str:
        """Extract the histology information.

        Useful for answering questions related to the histology of colorectal cancer patients
        
        Args:
            query (str): The query for the extraction purpose
        """
            # top_k (int): The number of results to retrieve
            # filter_key_list (List[str]): The list of filter keys
            # filter_value_list (List[str]): The list of filter values

        # exact_match_filters = [
        #     ExactMatchFilter(key=k, value=v)
        #     for k, v in zip(filter_key_list, filter_value_list)
        # ]
        # retriever = VectorIndexRetriever(
        #     self._index,
        #     filters=MetadataFilters(filters=exact_match_filters),
        #     top_k=top_k,
        # )
        # query_engine = RetrieverQueryEngine.from_args(retriever)
        retriever = self._index.as_retriever(similarity_top_k=self.similarity_top_k)
            # https://docs.llamaindex.ai/en/latest/examples/retrievers/auto_merging_retriever.html
        # retriever = AutoMergingRetriever(
        #     retriever, 
        #     self._index.storage_context, 
        #     verbose=True
        # )

        #query_engine = RetrieverQueryEngine.from_args(automerging_retriever, node_postprocessors=[rerank], 
         #                                                    service_context = self._context)
        query_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[self.rerank], 
                                                             service_context = self._context)

        update_query_engine_prompts(self.ccr_histology_template, query_engine)
        #prompts_dict = query_engine.get_prompts()
        #display_prompt_dict(prompts_dict)

        response = query_engine.query(query)        
        return str(response)