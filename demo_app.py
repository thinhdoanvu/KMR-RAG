import threading
import streamlit as st
import time
import numpy as np
from islab.nlp.cancer_registry.datasets.utils import CRCPlainTextReader
from pathlib import Path
from streamlit_timeline import timeline
from dateutil.parser import parse
from islab.nlp.cancer_registry.agents.llamaindex import CancerRegistryAgenetSystem
from streamlit_markmap import markmap

# use full page width
st.set_page_config(page_title="Cancer Registry Agent Demo", layout="wide")

tab_data, analysis_detail = st.tabs(["資料集", "分析細節"])

@st.cache_data
def load_dataset(path):
	reader = CRCPlainTextReader()
	patients = reader.load_data(path)	
	patient_list = {}
	for pid in patients.keys():	
		patient_list[Path(pid).stem]=[prepare_time_line_data(patients[pid]), patients[pid]]	
	return patient_list
	
def prepare_time_line_data(docs):
	tl_info = {}
	tl_info['events']=[]
	for doc in docs:
		report = {}
		date = parse(doc.metadata['date'])
		report["start_date"] = {"year": date.year, "month": date.month, "day": date.day}
		report["text"] = {
			"text": doc.text.replace('\n', '<br/>')
		}
		tl_info['events'].append(report)	
	return tl_info

def change_patient():
	with tab_data:
		if st.session_state.select_patient:
			timeline(patient_list[st.session_state.select_patient][0], height=900)

def log_function(mesg = None, idx = -1, cite = None, mindmap = None, append_mindmap = False, chat_msg= None, chat_role = 'system'):
	#with st.session_state['status_container'] as sc:
	if chat_msg is not None:
		with analysis_detail:
			with st.chat_message(chat_role):
				st.markdown(chat_msg)
	elif mindmap is not None:
		if not append_mindmap:			
			st.session_state['mindmap'] = '''
---
markmap:
  colorFreezeLevel: 2
---


'''
			st.session_state['mindmap'] = st.session_state['mindmap'] + mindmap + '\n\n'
		else:
			st.session_state['mindmap'] = st.session_state['mindmap'] + mindmap + '\n\n'
		with analysis_detail:
			markmap(st.session_state['mindmap'], height=400)

	elif idx < 0:
		st.write(mesg)
	else:
		status_text.info(mesg)

def initialize_system():
	doc_idx_llm, agent_llm, index_llm = initialize_llms()
	agent_system = CancerRegistryAgenetSystem(doc_idx_llm, agent_llm, index_llm)#st.session_state['doc_llm'], st.session_state['agent_llm'], st.session_state['idx_llm'], logger=log_function)					
	agent_system.set_logger(log_function)
	st.session_state['agent_system'] = agent_system	

@st.cache_resource
def initialize_llms():
	agent_llm = CancerRegistryAgenetSystem.create_agent_llm()
	index_llm = CancerRegistryAgenetSystem.create_index_agent_llm()
	doc_idx_llm = CancerRegistryAgenetSystem.create_doc_idx_llm()
	return doc_idx_llm, agent_llm, index_llm

with st.sidebar:
	status_text = st.sidebar.empty()

	if 'dataset' not in st.session_state:
		with st.spinner('載入資料集...'):
			patient_list = load_dataset('datasets/crc/testset')
			st.session_state['dataset'] = patient_list
	else:
		patient_list = st.session_state['dataset']
	status_text.success('資料載入完成!')
	
	if 'agent_system' not in st.session_state:
		with st.spinner('系統初始化中...'):
			agent_system = initialize_system()			
	else:
		agent_system = st.session_state['agent_system']
	status_text.success('系統初始化完成!')				
		
	option = st.selectbox(
				'選擇病患',
				patient_list.keys(), 
				on_change=change_patient, key='select_patient', index = None)
	st.write('目前選擇的病患:', option)
		
	with st.form(key='patient_selection'):
		query = st.text_input('您的查詢', 'What is the histology coding of the patient if he has colorectal cancer?')
		analyze = st.form_submit_button("分析")
		if analyze:
			if st.session_state.select_patient:
				status_text.info("分析中...")
				#if 'status_container' not in st.session_state:
					#status_container = st.sidebar.status('閒置中', state = 'complete')
					#st.session_state['status_container'] = status_container
				#status_container.empty()

				#with st.session_state['status_container'] as sc:
				with st.status('開始分析流程...', expanded = True) as sc:#st.session_state['status_container'] as sc:
					#sc.update(label='', state='running', expanded=True)
					st.write('建置代理人...')
					agent = agent_system.build_agents(patient_list[st.session_state.select_patient][1])
					st.write('代理人分析中...')
					results = agent.query(query)
					status_text.success(results.response)		
					sc.update(label='分析完成', state='complete', expanded=False)
			else:
				status_text.warning("請先選擇病患!")