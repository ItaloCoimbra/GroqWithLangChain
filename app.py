import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

GROQ_API_KEY = "CHAVEGROQAQUI"
CONTEXT_FILE = "contextoAberto.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def validate_environment():
    """Valida que as dependências e arquivos necessários estão configurados"""
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("A chave da API do Groq não está configurada.")
    if not os.path.exists(CONTEXT_FILE):
        raise FileNotFoundError(f"O arquivo '{CONTEXT_FILE}' não foi encontrado!")

def load_and_process_document():
    """Carrega e processa o documento manualmente"""
    print("Carregando e processando o documento de contexto...")
    try:
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_text(content)
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.from_texts(splits, embeddings)
        
        print("Documento processado com sucesso!")
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar ou processar o documento: {str(e)}")

def setup_qa_chain():
    """Configura a chain de QA"""
    try:
        chat = ChatGroq(
            temperature=0.7,
            model="mixtral-8x7b-32768"
        )
        template = """Você é um assistente amigável que responde perguntas baseado no contexto fornecido.
        Mantenha um tom conversacional e natural. SEMPRE DIGITE EM PORTUGUES, SEMPRE EM PORTUGUES, INDEPENDENTE.
        
        Contexto relevante do documento: {context}
        
        Humano: {question}
        
        Assistente:"""
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | chat | StrOutputParser()
    except Exception as e:
        raise RuntimeError(f"Erro ao configurar a chain de QA: {str(e)}")


def save_conversation(conversation_history):
    """Salva o histórico da conversa em um arquivo"""
    try:
        with open("historico_chat.txt", "w", encoding="utf-8") as f:
            f.write("=== Histórico do Chat ===\n\n")
            for i, interaction in enumerate(conversation_history, 1):
                f.write(f"--- Interação {i} ---\n")
                f.write(f"Você: {interaction['question']}\n")
                f.write(f"Assistente: {interaction['response']}\n\n")
        print("Conversa salva em 'historico_chat.txt'")
    except Exception as e:
        print(f"Erro ao salvar o histórico: {str(e)}")

def interactive_chat(vectorstore, qa_chain):
    """Gerencia a interação do chat"""
    print("\n=== Chat Iniciado ===")
    print("Digite 'sair' para encerrar o chat")
    print("Digite 'salvar' para salvar a conversa\n")
    
    conversation_history = []

    while True:
        question = input("\nVocê: ")
        if question.lower() == "sair":
            break
        elif question.lower() == "salvar":
            save_conversation(conversation_history)
            continue

        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            response = qa_chain.invoke({
                "context": context,
                "question": question
            })
            conversation_history.append({"question": question, "response": response})
            print("\nAssistente:", response)
            print("\n" + "-" * 50)
        except Exception as e:
            print(f"Erro durante a interação: {str(e)}")

def start_chat():
    """Inicia o chat interativo com o documento"""
    try:
        validate_environment()
        vectorstore = load_and_process_document()
        qa_chain = setup_qa_chain()
        interactive_chat(vectorstore, qa_chain)
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")

if __name__ == "__main__":
    start_chat()
