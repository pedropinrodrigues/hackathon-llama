import streamlit as st
import os
import time
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from crewai_tools import tool, ScrapeWebsiteTool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Carregar variáveis de ambiente
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

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
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Função para carregar PDFs
def carregar_pdfs(pasta="pdfs"):
    if not os.path.exists(pasta):
        st.error(f"Erro: A pasta '{pasta}' não foi encontrada.")
        return []
    
    arquivos_pdf = [os.path.join(pasta, f) for f in os.listdir(pasta) if f.lower().endswith('.pdf')]
    if arquivos_pdf:
        st.info(f"{len(arquivos_pdf)} arquivos PDF encontrados na pasta '{pasta}'.")
        return arquivos_pdf
    else:
        st.error(f"Erro: Nenhum arquivo PDF encontrado na pasta '{pasta}'.")
        return []

# Função de configuração do LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        api_key=groq_api_key
    )

# Função para buscar delegacias
@tool
def buscar_delegacias_proximas(endereco_busca: str) -> str:
    """
    Busca delegacias de polícia próximas a um endereço fornecido.

    Parâmetros:
    endereco_busca (str): Endereço para a busca de delegacias próximas.

    Retorno:
    str: Lista de delegacias encontradas ou uma mensagem de erro caso ocorra uma falha na requisição.
    """
    api_key = "9e65dad0b5342f127b90b31b40f4911f85d37019"
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": f"delegacia de polícia perto de {endereco_busca}", "gl": "br", "hl": "pt"}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        results = response.json()
        places = results.get('places', [])
        output = ""

        if places:
            for place in places:
                nome = place.get('title', 'Não disponível')
                endereco = place.get('address', 'Não disponível')
                cid = place.get('cid', 'Não disponível')

                if 'delegacia' in nome.lower() or 'polícia' in nome.lower() or "distrito policial" in nome.lower():
                    google_maps_link = f"https://www.google.com/maps?q={endereco.replace(' ', '+')}"
                    output += f"Nome: {nome}\nEndereço: {endereco}\nCID: {cid}\nLink do Google Maps: {google_maps_link}\n\n"
        else:
            output = "Nenhuma delegacia encontrada na região."
        
        return output
    
    except requests.exceptions.RequestException as e:
        return f"Erro na busca: {str(e)}"

# Função para executar o crew de localização
def executar_crew_localizacao(endereco):
    llm = get_llm()
    identificador = Agent(
        llm=llm,
        role="Identificador de Delegacias",
        goal=f"Localizar quais são as delegacias mais próximas de {endereco}",
        backstory=f"Conhece a região do endereço ({endereco}) muito bem e pode auxiliar a vítima",
        allow_delegation=False,
        verbose=True,
        tools=[buscar_delegacias_proximas]
    )
    escritor = Agent(
        llm=llm,
        role="Agente de Suporte de Vítimas",
        goal="Escrever uma mensagem empática indicando as delegacias mais próximas para que a vítima preste queixa",
        backstory="Psicólogo experiente, empático e assertivo em suas sugestões.",
        allow_delegation=False,
        verbose=True
    )
    identificar_delegacias = Task(description=f"Liste as delegacias mais próximas da vítima localizada em {endereco}.", expected_output="Uma lista com as localidades mais próximas.", agent=identificador)
    escrita = Task(description="Escreva uma resposta empática indicando as delegacias próximas da vítima.", expected_output="Mensagem listando as delegacias próximas.", agent=escritor)
    crew = Crew(agents=[identificador, escritor], tasks=[identificar_delegacias, escrita], verbose=2)

    return crew.kickoff(inputs={"endereco": endereco})

# Função para executar o crew de denúncia
def executar_crew_denuncia(victim_name, conversation_text):
    llm = get_llm()
    
    estrutura = Agent(
        llm=llm,
        role="Especialista em Estruturação de Documentos de Denúncia",
        goal="Estruturar um documento de denúncia de violência contra a mulher seguindo padrões técnicos e jurídicos",
        backstory=f"Especialista em documentação jurídica auxiliando {victim_name}.",
        allow_delegation=False,
        verbose=True
    )
    
    escritor = Agent(
        llm=llm,
        role="Redator Jurídico Especializado em Denúncias",
        goal="Transformar relatos e evidências em um documento formal",
        backstory=f"Redator jurídico ajudando {victim_name}.",
        allow_delegation=False,
        verbose=True
    )
    
    jurista = Agent(
        llm=llm,
        role="Especialista em Lei Maria da Penha",
        goal="Classificar violações e recomendar medidas legais apropriadas",
        backstory=f"Jurista especializado em violência de gênero analisando o caso de {victim_name}.",
        allow_delegation=False,
        verbose=True
    )

    analisar_violencia = Task(
        description="Analise o relato e classifique o ocorrido conforme a Lei Maria da Penha.",
        expected_output="Relatório jurídico da análise.",
        agent=jurista
    )
    
    planejamento = Task(
        description="Elabore um modelo de documento para denúncia de violência com base no relato.",
        expected_output="Plano de denúncia.",
        agent=estrutura
    )
    
    escrita = Task(
        description=f"Redija o documento de denúncia para {victim_name}.",
        expected_output="Documento de denúncia estruturado.",
        agent=escritor
    )
    
    crew = Crew(
        agents=[jurista, estrutura, escritor],
        tasks=[analisar_violencia, planejamento, escrita],
        verbose=2
    )

    # Correção: usar "victim_name" no dicionário de inputs
    return crew.kickoff(inputs={
        "victim_name": victim_name,
        "conversation": conversation_text
    })


# Configuração da interface com abas
tabs = st.tabs(["🚔 Localizar Delegacias", "📝 Criar Denúncia", "👮‍♀️ Assistente Lei Maria da Penha"])

# Aba 1: Localizar Delegacias
with tabs[0]:
    st.title("🚔 Localizador de Delegacias Próximas")
    st.markdown("### Encontre delegacias próximas à sua localização")
    with st.form("busca_form"):
        endereco = st.text_input("Digite seu endereço", placeholder="Ex: Rua MMDC, 80, Butantã, São Paulo")
        submitted_localizacao = st.form_submit_button("🔍 Buscar Delegacias")
    if submitted_localizacao and endereco:
        with st.spinner('Buscando delegacias próximas...'):
            try:
                resultado = executar_crew_localizacao(endereco)
                st.success("Busca realizada com sucesso!")
                st.markdown("### Resultado da Busca")
                st.write(resultado)
            except Exception as e:
                st.error(f"Ocorreu um erro durante a busca: {str(e)}")

# Aba 2: Criar Denúncia
with tabs[1]:
    st.title("📝 Assistente de Denúncia")
    st.markdown("### Auxílio na criação do documento de denúncia")
    with st.form("denuncia_form"):
        victim_name = st.text_input("Nome da Vítima", placeholder="Digite seu nome completo")
        conversation_text = st.text_area("Relato dos Acontecimentos", placeholder="Descreva detalhadamente os acontecimentos...", height=300)
        submitted_denuncia = st.form_submit_button("📋 Gerar Documento de Denúncia")
    if submitted_denuncia and victim_name and conversation_text:
        with st.spinner('Gerando documento de denúncia...'):
            try:
                resultado = executar_crew_denuncia(victim_name, conversation_text)
                st.success("Documento gerado com sucesso!")
                st.markdown("### Documento de Denúncia")
                st.write(resultado)
            except Exception as e:
                st.error(f"Ocorreu um erro durante a geração do documento: {str(e)}")

# Aba 3: Assistente Lei Maria da Penha
with tabs[2]:
    st.title("👮‍♀️ Assistente Virtual - Lei Maria da Penha")
    st.subheader("Tire suas dúvidas sobre a Lei Maria da Penha")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Olá! Eu sou seu Assistente Virtual sobre a Lei Maria da Penha. Aconteceu algo com que posso ajudar?"}]
    
    arquivos_pdf = carregar_pdfs()
    
    if arquivos_pdf:
            try:
                # Configuração inicial do chat
                if "chat_initialized" not in st.session_state:
                    with st.spinner("Inicializando o assistente..."):
                        model_name = "sentence-transformers/all-MiniLM-L6-v2"
                        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
                        
                        # Carregar e dividir documentos de todos os PDFs
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
                        
                        # Configurar LLM e prompt
                        llm = ChatGroq(
                            groq_api_key=groq_api_key,
                            model_name="llama3-70b-8192",
                            temperature=0.1
                        )

                        prompt = ChatPromptTemplate.from_template("""
                        Você é uma assistente especializada na Lei Maria da Penha (Lei nº 11.340/2006) e deseja entender o máximo de detalhes sobre a situação da pessoa que está em contato com você.
                        Responda à pergunta de maneira empática e faça perguntas que ajudem a entender melhor a situação da vítima, sempre buscando obter mais detalhes e mostrando compreensão.

                        Algumas diretrizes para sua resposta:
                        - Use linguagem acessível, evitando jargões jurídicos complexos
                        - Se a situação envolver risco, pergunte se a pessoa já buscou ajuda e ofereça informações sobre canais de apoio
                        - Mantenha-se estritamente dentro do conteúdo da lei fornecido no contexto
                        - Finalize cada resposta com uma pergunta, incentivando a continuidade da conversa e a obtenção de mais detalhes sobre a situação. Faça apenas uma pergunta por mensagem.
                        - Antes de perguntar, ajude ela com as medidas legais com base na lei maria da penha.

                        <contexto>
                        {context}
                        </contexto>

                        Pergunta: {input}
                        Histórico da Conversa: {history}
                        """)


                        # Criar chains
                        document_chain = create_stuff_documents_chain(llm, prompt)
                        retriever = vectors.as_retriever(search_kwargs={"k": 4})
                        st.session_state.retrieval_chain = create_retrieval_chain(retriever, document_chain)
                        st.session_state.chat_initialized = True

                # Display chat messages
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])

                # Chat input
                if pergunta := st.chat_input("Digite sua pergunta sobre a Lei Maria da Penha"):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": pergunta})
                    with st.chat_message("user"):
                        st.write(pergunta)

                    # Gerar histórico da conversa
                    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        with st.spinner("Analisando sua pergunta..."):
                            start = time.time()
                            response = st.session_state.retrieval_chain.invoke({"input": pergunta, "history": history})
                            tempo = time.time() - start
                            
                            st.write(response['answer'])
                            st.caption(f"Tempo de resposta: {tempo:.2f} segundos")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})

            except Exception as e:
                st.error(f"Ocorreu um erro ao processar sua solicitação: {str(e)}")


# Rodapé
st.markdown("---")
st.markdown("Desenvolvido para auxiliar vítimas de violência. Ligue 180 para a Central de Atendimento à Mulher.", help="Este é um serviço de utilidade pública")
