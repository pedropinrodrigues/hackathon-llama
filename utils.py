# utils.py

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
import streamlit as st

# Carregar variáveis de ambiente
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

# Função para carregar PDFs
def carregar_pdfs(pasta="pdfs"):
    if not os.path.exists(pasta):
        st.error(f"Erro: A pasta '{pasta}' não foi encontrada.")
        return []
    
    arquivos_pdf = [os.path.join(pasta, f) for f in os.listdir(pasta) if f.lower().endswith('.pdf')]
    if arquivos_pdf:
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
    api_key = os.environ.get("SERPER_API_KEY")
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
        goal="Escrever uma mensagem super direta indicando as delegacias mais próximas para que a vítima preste queixa",
        backstory="Pessoa muito direta ao ponto, não gosta de enrolação",
        allow_delegation=False,
        verbose=True
    )
    identificar_delegacias = Task(description=f"Liste as delegacias mais próximas da vítima localizada em {endereco}.", expected_output="Uma lista com as localidades mais próximas.", agent=identificador)
    escrita = Task(description="Escreva uma mensagem recomendando que a pessoa preste queixa e liste as delegacias mais próximas", expected_output="Uma lista de delegacias, e uma recomendacao inicial para prestacação de queixa", agent=escritor)
    crew = Crew(agents=[identificador, escritor], tasks=[identificar_delegacias, escrita], verbose=2)

    return crew.kickoff(inputs={"endereco": endereco})

# Função para executar o crew de denúncia
def executar_crew_denuncia(victim_name, conversation_text):
    llm = get_llm()
    
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
        backstory=f"Jurista especializado em violência de gênero há 15 anos.",
        allow_delegation=False,
        verbose=True
    )

    analisar_violencia = Task(
        description="Analise o relato ({conversation}) e classifique o ocorrido conforme a Lei Maria da Penha.",
        expected_output="Relatório jurídico da análise.",
        agent=jurista
    )

    escrita = Task(
        description="""
        Analise o relato e crie um relatório detalhado que inclua:
        1. Resumo da situação relatada
        2. Descrição dos Acontecimentos e legislação violada
        3. Medidas protetivas, direitos a serem reinvindicados
        com base em {conversation} e no relatório jurídico
        Use o seguinte formato:
        - Informações Gerais
        - Detalhes do Caso
        - Leis Infringidas
        - Medidas Protetivas
        - Observações Adicionais
        """,
        expected_output="Relatório estruturado com todas as seções solicitadas.",
        agent=escritor
    )
    
    crew = Crew(
        agents=[jurista, escritor],
        tasks=[analisar_violencia, escrita],
        verbose=2
    )

    return crew.kickoff(inputs={
        "victim_name": victim_name,
        "conversation": conversation_text
    })

# Função para executar o crew de geração de relatórios com histórico da conversa
def executar_crew_relatorio(chat_history):
    llm = get_llm()

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
    analisar_violencia = Task(
        description="Analise a conversa {chat_history} e, conforme a lei Maria da Penha, classifique as violações e recomende medidas protetivas.",
        expected_output="Relatório jurídico com classificação de violações e medidas legais recomendadas",
        agent=jurista
    )

    crew = Crew(
        agents=[jurista, agente_relatorio],
        tasks=[analisar_violencia, tarefa_relatorio],
        verbose=2
    )

    return crew.kickoff(inputs={"chat_history": chat_history})

# Função para gerar conteúdo PDF
def gerar_pdf_conteudo(conteudo, titulo="Dossiê de Denúncia de Violência Doméstica", autor="Sistema"):
    buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    largura, altura = A4
    c = canvas.Canvas(buffer.name, pagesize=A4)
    
    cor_principal = colors.HexColor('#5e17eb')
    cor_secundaria = colors.HexColor('#718096')
    
    def adicionar_cabecalho():
        c.setFillColor(cor_principal)
        c.rect(0, altura - 80, largura, 80, fill=True)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, altura - 50, titulo)
        data_atual = datetime.now().strftime("%d/%m/%Y")
        c.setFont("Helvetica", 10)
        c.drawString(largura - 150, altura - 50, f"Data: {data_atual}")
    
    def adicionar_rodape():
        c.setStrokeColor(cor_secundaria)
        c.line(50, 50, largura - 50, 50)
        c.setFillColor(cor_secundaria)
        c.setFont("Helvetica", 8)
        c.drawString(50, 35, f"Autor: {autor}")
        c.drawString(largura - 150, 35, f"Página 1")
    
    def formatar_conteudo():
        styles = getSampleStyleSheet()
        estilo_normal = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            textColor=colors.black,
            spaceAfter=10
        )
        frame = Frame(50, 70, largura - 100, altura - 160, leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)
        story = [Paragraph(paragrafo.replace('\n', '<br/>'), estilo_normal) for paragrafo in conteudo.split('\n\n') if paragrafo.strip()]
        frame.addFromList(story, c)
    
    adicionar_cabecalho()
    formatar_conteudo()
    adicionar_rodape()
    c.save()
    buffer.seek(0)
    
    with open(buffer.name, "rb") as f:
        pdf_data = f.read()
    
    return pdf_data
