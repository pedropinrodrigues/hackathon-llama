# app.py
import os
import requests
import tempfile
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, Frame
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from crewai_tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import streamlit as st
import streamlit as st
from utils import (
    carregar_pdfs,
    get_llm,
    buscar_delegacias_proximas,
    executar_crew_localizacao,
    executar_crew_denuncia,
    executar_crew_relatorio,
    gerar_pdf_conteudo
)
import os
import time

# Configuração da página
st.set_page_config(
    page_title="Sistema de Suporte à Vítima e Assistente Jurídico",
    page_icon="⚖️",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f0f2f6;
        }
        .big-font {
            font-size:20px !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Inicialize o estado da página se ele não existir
if "page" not in st.session_state:
    st.session_state["page"] = "Assistente Lei Maria da Penha"

# Função para definir a página atual com base no botão clicado
def set_page(page_name):
    st.session_state["page"] = page_name

# Botões para navegar entre as páginas
st.sidebar.title("maria.ai")
st.sidebar.button("Falar com Maria", on_click=set_page, args=("Assistente Lei Maria da Penha",), use_container_width=True)
st.sidebar.button("Criar Denúncia", on_click=set_page, args=("Criar Denúncia",), use_container_width=True)
st.sidebar.button("Localizar Delegacias", on_click=set_page, args=("Localizar Delegacias",), use_container_width=True)

# Main content area based on selection
if st.session_state["page"] == "Localizar Delegacias":
    st.title("Localizador de Delegacias Próximas")
    st.markdown("### Encontre delegacias próximas à sua localização")
    with st.form("busca_form"):
        endereco = st.text_input("Digite seu endereço", placeholder="Ex: Rua MMDC, 80, Butantã, São Paulo")
        submitted_localizacao = st.form_submit_button("Buscar Delegacias", type="primary")
    if submitted_localizacao and endereco:
        with st.spinner('Buscando delegacias próximas...'):
            try:
                resultado = executar_crew_localizacao(endereco)
                st.success("Busca realizada com sucesso!")
                st.markdown("### Resultado da Busca")
                st.write(resultado)
            except Exception as e:
                st.error(f"Ocorreu um erro durante a busca: {str(e)}")

import streamlit as st

# Página de criação de denúncia
if st.session_state["page"] == "Criar Denúncia":
    # Título da Página
    st.markdown("<h1 style='color: #FF6F61;'>Assistente de Denúncia</h1>", unsafe_allow_html=True)

    # Caixa de informações iniciais
    st.write("### Bem-vinda ao Assistente de Denúncia")
    st.info(
        "Aqui, você pode gerar um relatório da conversa com o assistente ou criar um documento de denúncia preenchendo um formulário detalhado."
    )

    # Seção 1: Gerar Relatório do Histórico de Conversa
    st.markdown("<hr style='border:1px solid #f0f2f6;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #FF6F61;'>Gerar Relatório da Conversa</h2>", unsafe_allow_html=True)

    with st.container():
        # Colunas para conteúdo explicativo e botão
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(
                "Crie um relatório detalhado da sua conversa com o Assistente Virtual. "
                "Este documento será essencial para formalizar sua denúncia e coletar informações importantes sobre o histórico de violência."
            )
        with col2:
            if st.button("Gerar Relatório da Conversa"):
                if "messages" in st.session_state and len(st.session_state.messages) > 1:
                    with st.spinner('Gerando o relatório...'):
                        try:
                            chat_history = "\n".join([
                                f"{msg['role'].upper()}: {msg['content']}"
                                for msg in st.session_state.messages
                            ])
                            resultado = executar_crew_relatorio(chat_history)
                            pdf_data = gerar_pdf_conteudo(resultado)
                            st.success("Relatório gerado com sucesso!")
                            st.download_button(
                                label="📥 Baixar Relatório (PDF)",
                                data=pdf_data,
                                file_name="Relatorio_da_Conversa.pdf",
                                mime="application/pdf"
                            )
                        except Exception as e:
                            st.error(f"Erro na geração do relatório: {str(e)}")
                else:
                    st.warning("Histórico de conversa não disponível. Use o Assistente Virtual primeiro.")

    # Seção 2: Formulário para novo relato
    st.markdown("<hr style='border:1px solid #f0f2f6;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #FF6F61;'>Criar Nova Denúncia</h2>", unsafe_allow_html=True)

    # Caixa de texto e formulário em duas colunas
    with st.container():
        st.write(
            "Caso prefira, descreva diretamente os acontecimentos para gerar um documento de denúncia. "
            "Este documento ajudará a formalizar sua denúncia e fornecerá detalhes necessários para as autoridades."
        )

        # Layout do formulário com colunas
        

        st.write("#### Informe os Detalhes")
        st.info(
            "Preencha todos os campos abaixo para gerar o documento de denúncia com as informações necessárias."
        )

        # Variável para armazenar o PDF gerado
        pdf_data = None

        # Formulário
        with st.form("denuncia_form"):
            victim_name = st.text_input("Nome da Vítima", placeholder="Digite seu nome completo")
            conversation_text = st.text_area(
                "Relato dos Acontecimentos",
                placeholder="Descreva detalhadamente o que aconteceu, incluindo datas, locais, e pessoas envolvidas.",
                height=300
            )
            submitted_denuncia = st.form_submit_button("Gerar Documento de Denúncia")

        # Processamento do formulário fora do bloco st.form
        if submitted_denuncia:
            if victim_name and conversation_text:
                with st.spinner('Gerando documento de denúncia...'):
                    try:
                        resultado = executar_crew_denuncia(victim_name, conversation_text)
                        pdf_data = gerar_pdf_conteudo(resultado)
                        st.success("Documento de denúncia gerado com sucesso!")
                    except Exception as e:
                        st.error(f"Erro na geração do documento: {str(e)}")
            else:
                st.warning("Por favor, preencha todos os campos necessários.")
        
        # Exibir botão de download fora do formulário após o PDF ser gerado
        if pdf_data:
            st.download_button(
                label="📥 Baixar Documento de Denúncia (PDF)",
                data=pdf_data,
                file_name="Documento_de_Denuncia.pdf",
                mime="application/pdf"
            )
elif st.session_state["page"] == "Assistente Lei Maria da Penha":
    st.title("Suporte Virtual - Fale com Maria")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Olá! Sou Maria, sua assistente virtual, aqui para apoiar na luta contra a violência contra a mulher. Em que posso ajudar você?"}]
    
    arquivos_pdf = carregar_pdfs()
    
    if arquivos_pdf:
        try:
            if "chat_initialized" not in st.session_state:
                with st.spinner("Inicializando o assistente..."):
                    model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
                    
                    final_documents = []
                    for arquivo_pdf in arquivos_pdf:
                        loader = PyPDFLoader(arquivo_pdf)
                        docs = loader.load()
                        
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "]
                        )
                        final_documents.extend(text_splitter.split_documents(docs))
                    
                    vectors = FAISS.from_documents(final_documents, embeddings)
                    
                    llm = get_llm()
                    prompt = ChatPromptTemplate.from_template("""
                        Você é uma assistente especializada na Lei Maria da Penha (Lei nº 11.340/2006) e deseja entender o máximo de detalhes sobre a situação da pessoa que está em contato com você.
                        Responda à pergunta de maneira empática e faça perguntas que ajudem a entender melhor a situação da vítima, sempre buscando obter mais detalhes e mostrando compreensão.

                        Algumas diretrizes para sua resposta:
                        - Use linguagem acessível, evitando jargões jurídicos complexos
                        - Se a situação envolver risco, pergunte se a pessoa já buscou ajuda e ofereça informações sobre canais de apoio
                        - Mantenha-se estritamente dentro do conteúdo da lei fornecido no contexto
                        - Finalize cada resposta com uma pergunta, incentivando a continuidade da conversa e a obtenção de mais detalhes sobre a situação. Faça apenas uma pergunta por mensagem.
                        - Antes de perguntar, ajude ela com as medidas legais com base na lei maria da penha.
                        Caso você identifique que há necessidade de acionar as autoridades, fale para o usuário sobre a existencia da aba "localizar delegacias" e "criar denúncia" para que ele possa ir ao local e ter os documentos necessários para isso.
                        <contexto>
                        {context}
                        </contexto>
                        Pergunta: {input}
                        Histórico da Conversa: {history}""")

                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retriever = vectors.as_retriever(search_kwargs={"k": 4})
                    st.session_state.retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    st.session_state.chat_initialized = True

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if pergunta := st.chat_input("Digite sua pergunta sobre a Lei Maria da Penha"):
                st.session_state.messages.append({"role": "user", "content": pergunta})
                with st.chat_message("user"):
                    st.write(pergunta)

                history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                st.session_state['history'] = history
                
                with st.chat_message("assistant"):
                    with st.spinner("Analisando sua pergunta..."):
                        start = time.time()
                        response = st.session_state.retrieval_chain.invoke({"input": pergunta, "history": history})
                        tempo = time.time() - start
                        
                        st.write(response['answer'])
                        st.caption(f"Tempo de resposta: {tempo:.2f} segundos")
                
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar sua solicitação: {str(e)}")
