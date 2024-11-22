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
        template = """
        {{
            "result": [
                "Analise as respostas à minha pergunta em até 150 palavras.",
                "Estruture a análise nas seguintes seções, e **garanta que o HTML gerado esteja perfeito**:",
                "  <h2 class=\\"analise-titulo\\">Aspectos Positivos</h2>",
                "  <p class=\\"analise-texto\\">Descreva os aspectos positivos ou pontos fortes destacados nas respostas.</p>",
                "  <h2 class=\\"analise-titulo\\">Oportunidades de Melhoria</h2>",
                "  <p class=\\"analise-texto\\">Identifique áreas para melhoria ou feedback construtivo com base nas respostas.</p>",
                "  <h2 class=\\"analise-titulo\\">Impressões Gerais</h2>",
                "  <p class=\\"analise-texto\\">Forneça suas impressões gerais e um resumo das respostas.</p>",
                "Forneça **somente** as seções solicitadas, sem qualquer introdução, explicação adicional ou formatação fora do HTML. Não adicione nenhuma explicação como 'Aqui está a análise em formato HTML'."
            ],
            "prompt": {{
                "content": "{{context}}",
                "instructions": [
                    "Forneça **somente** as seções solicitadas, sem qualquer introdução ou explicação adicional, como 'Here is the analysis in HTML format'.",
                    "A análise deve ser estruturada nas seções solicitadas, utilizando apenas HTML e sem texto fora das tags HTML.",
                    "O **único formato aceito** para a análise é HTML. Não deve haver texto fora das tags HTML fornecidas.",
                    "Cada seção deve ter um título dentro da tag <h2> com a classe 'analise-titulo' e o conteúdo dentro da tag <p> com a classe 'analise-texto'.",
                    "Todas as tags HTML devem ser fechadas corretamente e sem erros de sintaxe.",
                    "A análise deve ser **precisa e fiel** ao formato solicitado, sem incluir qualquer formatação extra, explicações adicionais ou texto de introdução.",
                    "O HTML gerado deve ser **limpo**, **bem estruturado** e **sem falhas**, com o uso correto das tags <h2> e <p> conforme solicitado."
                ]
            }}
        }}
        """
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
