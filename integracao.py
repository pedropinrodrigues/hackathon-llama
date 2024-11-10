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
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile
import tempfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, Frame
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime

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

# Função para executar o crew de geração de relatórios com histórico da conversa
def executar_crew_relatorio(chat_history):
    llm = get_llm()
    print(chat_history)

    jurista = Agent(
        llm=llm,
        role="Especialista em Lei Maria da Penha",
        goal="Classificar violações e recomendar medidas legais apropriadas",
        backstory=f"Jurista especializado em violência de gênero há 15 anos.",
        allow_delegation=False,
        verbose=True
    )

    agente_relatorio = Agent(
        llm=llm,
        role="Agente de Geração de Relatórios",
        goal="Criar um relatório detalhado e estruturado com base no histórico de conversa do chat {chat_history}",
        backstory="Especialista em análise de conversas e geração de relatórios técnicos, com foco em casos de violência contra a mulher.",
        allow_delegation=False,
        verbose=True
    )

    tarefa_relatorio = Task(
        description="""
        Analise o histórico da conversa e crie um relatório detalhado que inclua:
        1. Resumo da situação relatada
        2. Descrição dos Acontecimentos e legislação violada
        3. Medidas protetivas, direitos a serem reinvindicados
        com base em {chat_history} e no relatório jurídico
        Use o seguinte formato:
        - Informações Gerais
        - Detalhes do Caso
        - Leis Infringidas
        - Medidas Protetivas
        - Observações Adicionais
        """,
        expected_output="Relatório estruturado com todas as seções solicitadas.",
        agent=agente_relatorio
    )
    analisar_violencia = Task(
        description="Analise a conversa {chat_history} e, conforme a lei maria da penha, classifique as violações e recomende medidas protetivas legais que devem ser reinvindicadas",
        expected_output="Relatório jurídico com classificação de violações e medidas legais recomendadas",
        agent=jurista
    )

    crew = Crew(
        agents=[jurista, agente_relatorio],
        tasks=[analisar_violencia, tarefa_relatorio],
        verbose=2
    )

    return crew.kickoff(inputs={"chat_history": chat_history})


# Configuração da interface com abas
tabs = st.tabs([ "👮‍♀️ Assistente Lei Maria da Penha","📝 Criar Denúncia","🚔 Localizar Delegacias"])

# Inicializa 'history' no session_state, se ainda não estiver definido
if 'history' not in st.session_state:
    st.session_state['history'] = ""

# Aba 1: Localizar Delegacias
with tabs[2]:
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
def gerar_pdf_conteudo(conteudo, titulo="Dossiê de Denúncia de Violência Doméstica", autor="Sistema"):
    # Criar buffer temporário
    buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    
    # Configurações da página
    largura, altura = A4
    c = canvas.Canvas(buffer.name, pagesize=A4)
    
    # Cores
    cor_principal = colors.HexColor('#1a365d')  # Azul escuro
    cor_secundaria = colors.HexColor('#718096')  # Cinza
    
    def adicionar_cabecalho():
        # Retângulo do cabeçalho
        c.setFillColor(cor_principal)
        c.rect(0, altura - 80, largura, 80, fill=True)
        
        # Título
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, altura - 50, titulo)
        
        # Data
        data_atual = datetime.now().strftime("%d/%m/%Y")
        c.setFont("Helvetica", 10)
        c.drawString(largura - 150, altura - 50, f"Data: {data_atual}")
    
    def adicionar_rodape():
        # Linha do rodapé
        c.setStrokeColor(cor_secundaria)
        c.line(50, 50, largura - 50, 50)
        
        # Texto do rodapé
        c.setFillColor(cor_secundaria)
        c.setFont("Helvetica", 8)
        c.drawString(50, 35, f"Autor: {autor}")
        c.drawString(largura - 150, 35, f"Página 1")
    
    def formatar_conteudo():
        # Estilo para o conteúdo
        styles = getSampleStyleSheet()
        estilo_normal = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            textColor=colors.black,
            spaceAfter=10
        )
        
        # Criar frame para o conteúdo
        frame = Frame(
            50,  # x
            70,  # y (acima do rodapé)
            largura - 100,  # largura
            altura - 160,  # altura (abaixo do cabeçalho)
            leftPadding=0,
            bottomPadding=0,
            rightPadding=0,
            topPadding=0
        )
        
        # Processar o conteúdo
        story = []
        for paragrafo in conteudo.split('\n\n'):
            if paragrafo.strip():
                p = Paragraph(paragrafo.replace('\n', '<br/>'), estilo_normal)
                story.append(p)
        
        # Desenhar o conteúdo
        frame.addFromList(story, c)
    
    # Desenhar elementos
    adicionar_cabecalho()
    formatar_conteudo()
    adicionar_rodape()
    
    # Finalizar PDF
    c.save()
    buffer.seek(0)
    
    # Ler e retornar o conteúdo binário
    with open(buffer.name, "rb") as f:
        pdf_data = f.read()
    
    return pdf_data

# Atualize o código da aba de denúncia
with tabs[1]:
    st.title("📝 Assistente de Denúncia")

    # Seção 1: Gerar relatório baseado no chat
    with st.container():
        st.markdown("### 📊 Gerar Relatório do Histórico de Conversa")
        st.info("""
            Use esta opção para gerar um relatório baseado na sua conversa com o Assistente Virtual.
            O relatório incluirá todo o histórico de interação e orientações recebidas.
        """)
        
        # Botão para gerar relatório do chat
        if st.button("📊 Gerar Relatório da Conversa", 
                    type="primary",
                    use_container_width=True):
            if "messages" in st.session_state and len(st.session_state.messages) > 1:
                with st.spinner('Gerando relatório baseado no histórico do chat...'):
                    try:
                        chat_history = "\n".join([
                            f"{msg['role'].upper()}: {msg['content']}"
                            for msg in st.session_state.messages
                        ])
                        resultado = executar_crew_relatorio(chat_history)
                        pdf_data = gerar_pdf_conteudo(resultado)
                        
                        st.success("✅ Relatório gerado com sucesso!")
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.download_button(
                                label="📥 Baixar Relatório da Conversa (PDF)",
                                data=pdf_data,
                                file_name="Relatorio_da_Conversa.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"❌ Erro na geração do relatório: {str(e)}")
            else:
                st.warning("⚠️ Não há histórico de conversa disponível. Por favor, utilize primeiro o Assistente Virtual na aba 'Assistente Lei Maria da Penha'.")

    # Separador visual
    st.divider()

    # Seção 2: Formulário para novo relato
    with st.container():
        st.markdown("### 📋 Criar Nova Denúncia")
        st.info("""
            Use esta opção para criar um novo documento de denúncia, 
            relatando detalhadamente os fatos ocorridos.
        """)
        
        with st.form("denuncia_form"):
            victim_name = st.text_input(
                "Nome da Vítima",
                placeholder="Digite seu nome completo"
            )
            
            conversation_text = st.text_area(
                "Relato dos Acontecimentos",
                placeholder="Descreva detalhadamente o que aconteceu. Inclua informações como data, local, " 
                          "pessoas envolvidas e qualquer outro detalhe relevante...",
                height=300
            )
            
            submitted_denuncia = st.form_submit_button(
                "📋 Gerar Documento de Denúncia",
                use_container_width=True,
                type="primary"
            )

        # Processamento do formulário de denúncia
        if submitted_denuncia:
            if victim_name and conversation_text:
                with st.spinner('Gerando documento de denúncia...'):
                    try:
                        resultado = executar_crew_denuncia(victim_name, conversation_text)
                        pdf_data = gerar_pdf_conteudo(resultado)
                        
                        st.success("✅ Documento de denúncia gerado com sucesso!")
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.download_button(
                                label="📥 Baixar Documento de Denúncia (PDF)",
                                data=pdf_data,
                                file_name="Documento_de_Denuncia.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"❌ Erro na geração do documento: {str(e)}")
            else:
                st.warning("⚠️ Por favor, preencha todos os campos necessários.")
# Aba 3: Assistente Lei Maria da Penha
with tabs[0]:
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

                    # Atualizar histórico da conversa
                    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    st.session_state['history'] = history
                    
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
