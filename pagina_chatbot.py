import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

def carregar_pdfs(pasta="pdfs"):
    """Carrega todos os arquivos PDF encontrados na pasta especificada."""
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

def initialize_chat_elements():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Olá! Eu sou seu Assistente Virtual, especializado na Lei Maria da Penha. Aconteceu alguma coisa?"}
        ]

def create_sidebar():
    with st.sidebar:
        st.title("Sobre o Assistente")
        st.markdown("""
        Este é um assistente virtual especializado na Lei Maria da Penha (Lei nº 11.340/2006).
        
        ### Canais de Ajuda
        - **Ligue 180**: Central de Atendimento à Mulher
        - **Ligue 190**: Polícia Militar
        
        ### Como usar
        Digite sua dúvida sobre a Lei Maria da Penha no campo de texto abaixo e receba orientações baseadas na legislação.
        
        ### Importante
        Em caso de emergência, procure ajuda imediatamente através dos números acima.
        """)

def main():
    st.set_page_config(
        page_title="Assistente Virtual - Lei Maria da Penha",
        page_icon="👮‍♀️",
        layout="wide"
    )
    
    st.title("Assistente Virtual - Lei Maria da Penha")
    st.subheader("Tire suas dúvidas sobre a Lei Maria da Penha")
    
    create_sidebar()
    initialize_chat_elements()
    
    # Carregar todos os arquivos PDF automaticamente
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
                    Você é um assistente especializado na Lei Maria da Penha (Lei nº 11.340/2006). 
                    Responda à pergunta com base apenas no contexto fornecido da lei.
                    Seja claro, objetivo e empático em suas respostas.
                    
                    Algumas diretrizes para sua resposta:
                    - Use linguagem acessível, evitando jargões jurídicos complexos
                    - Se a pergunta envolver situação de risco, inclua informações sobre canais de ajuda
                    - Mantenha-se estritamente dentro do conteúdo da lei fornecido no contexto
                    - Ao final de cada mensagem, voce deve enviar uma mensagem, seguindo o roteiro.

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

if __name__ == "__main__":
    main()
