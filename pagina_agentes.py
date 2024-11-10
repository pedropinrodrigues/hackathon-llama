import streamlit as st
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from crewai_tools import tool, ScrapeWebsiteTool
import requests

# Configuração da página
st.set_page_config(
    page_title="Sistema de Suporte à Vítima",
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

# Configuração do LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        api_key='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'
    )

# Função para buscar delegacias
@tool
def buscar_delegacias_proximas(endereco_busca: str) -> str:
    """Busca delegacias de polícia próximas a um endereço fornecido."""
    api_key = "9e65dad0b5342f127b90b31b40f4911f85d37019"
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "q": f"delegacia de polícia perto de {endereco_busca}",
        "gl": "br",
        "hl": "pt"
    }

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
                    
                    output += f"Nome: {nome}\n"
                    output += f"Endereço: {endereco}\n"
                    output += f"CID: {cid}\n"
                    output += f"Link do Google Maps: {google_maps_link}\n\n"
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

    identificar_delegacias = Task(
        description=f"Liste as delegacias mais próximas da vítima localizada em {endereco} passe endereco e link.",
        expected_output="Uma lista com as localidades mais próximas da posição da vítima.",
        agent=identificador
    )

    escrita = Task(
        description="Escreva uma resposta de chatbot de forma empática, indicando as delegacias próximas da vítima escreva endereco e link.",
        expected_output="Uma mensagem direta e empática listando as delegacias próximas da vítima.",
        agent=escritor
    )

    crew = Crew(
        agents=[identificador, escritor],
        tasks=[identificar_delegacias, escrita],
        verbose=2
    )

    return crew.kickoff(inputs={"endereco": endereco})

# Função para executar o crew de denúncia
def executar_crew_denuncia(victim_name, conversation_text):
    llm = get_llm()

    estrutura = Agent(
        llm=llm,
        role="Especialista em Estruturação de Documentos de Denúncia",
        goal="Estruturar um documento de denúncia de violência contra a mulher seguindo padrões técnicos e jurídicos",
        backstory="""Você é um especialista em documentação jurídica e casos violência doméstica. 
        Você está auxiliando {victim} a organizar sua denúncia de forma profissional.
        """,
        allow_delegation=False,
        verbose=True
    )


    escritor = Agent(
    llm=llm,
    role="Redator Jurídico Especializado em Denúncias",
    goal="Transformar relatos e evidências em um documento formal",
    backstory="""Você é redator jurídico especializado em violência doméstica, 
    ajudando {victim} a documentar sua experiência de forma clara e correta.
    """,
    allow_delegation=False,
    verbose=True
    )



    jurista = Agent(
    llm=llm,
    role="Especialista em Lei Maria da Penha",
    goal="Classificar violações e recomendar medidas legais apropriadas",
    backstory="""Você é jurista especializado em violência de gênero e na lei maria da penha, analisando o caso de {victim} para garantir a aplicação correta da lei.
    """,
    allow_delegation=False,
    verbose=True
    )


    planejamento = Task(
    description="""Com base apenas no relato da vitima, elabore um modelo de documento para denúncia da violencia ocorrida, esse documento deve ser uma dossie das situações descritas pela vitima. Tudo presente na estrutura, deve vir de {conversation}.""",
    expected_output="""Plano de denuncia de violencia contra mulher""",
    agent=estrutura,
    )


    escrita = Task(
    description="""
    Com base na estrutura fornecida e em {conversation}
    1. Redija o documento de denúncia para {victim}
    3. Mantenha linguagem técnica
    4. Destaque medidas protetivas urgentes
    NENHUMA INFORMAÇÃO DEVE SER INVENTADA, BASEIE-SE APENAS NO RELATORIO DE ANALISE DE VIOLENCIA E EM {conversation}. 
    EM NENHUMA HIPÓTESE ESCREVA CAMPOS PARA SEREM PREENCHIDOS, TUDO ESCRITO DEVE PARTIR DE {conversation} ou da analise de violencia
    """,
    expected_output="""Documento de denúncia completo, bem estruturado e fiel ao relato da vítima""",
    agent=escritor,
    )

    analisar_violencia = Task(
    description="""
    Com base no relato fornecido:
    Analise e classifique o ocorrido com base na lei maria da penha
    faca isso com base em {conversation}
    """,
    expected_output="""Análise jurídica com:
    relaorio relacionando os casos destacados com a lei maria da penha, de forma a garantir que todos os direitos das mulheres serão cobertos.
    """,
    agent=jurista,
    )




    crew = Crew(
        agents=[jurista, estrutura, escritor],
        tasks=[analisar_violencia, planejamento, escrita],
        verbose=2
    )

    return crew.kickoff(inputs={
        "victim": victim_name,
        "conversation": conversation_text
    })

# Interface principal com tabs
tabs = st.tabs(["🚔 Localizar Delegacias", "📝 Criar Denúncia"])

# Tab 1: Localizar Delegacias
with tabs[0]:
    st.title("🚔 Localizador de Delegacias Próximas")
    st.markdown("### Encontre delegacias próximas à sua localização")
    
    with st.form("busca_form"):
        endereco = st.text_input(
            "Digite seu endereço",
            placeholder="Ex: Rua MMDC, 80, Butantã, São Paulo"
        )
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

# Tab 2: Criar Denúncia
with tabs[1]:
    st.title("📝 Assistente de Denúncia")
    st.markdown("### Auxílio na criação do documento de denúncia")
    
    with st.form("denuncia_form"):
        victim_name = st.text_input(
            "Nome da Vítima",
            placeholder="Digite seu nome completo"
        )
        conversation_text = st.text_area(
            "Relato dos Acontecimentos",
            placeholder="Descreva detalhadamente os acontecimentos, incluindo datas e eventos importantes...",
            height=300
        )
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

# Informações adicionais
with st.expander("ℹ️ Sobre o Sistema"):
    st.write("""
    Este sistema oferece dois serviços principais:
    1. **Localização de Delegacias**: Encontre delegacias próximas à sua localização
    2. **Assistente de Denúncia**: Auxílio na criação de documentos formais de denúncia
    
    Em caso de emergência, ligue 190 ou procure a delegacia mais próxima imediatamente.
    """)

# Footer
st.markdown("---")
st.markdown(
    "Desenvolvido para auxiliar vítimas de violência. Ligue 180 para a Central de Atendimento à Mulher.",
    help="Este é um serviço de utilidade pública"
)