import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
import time

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
    """Configura a chain de QA com o conteúdo já definido diretamente no prompt."""
    try:
        chat = ChatGroq(
            temperature=0.7,
            model="llama-3.1-8b-instant"
        )
        
        # Prompt com o conteúdo final diretamente
        template = """
        Gere perguntas criativas para uma pesquisa do tipo satisfação, com as seguintes especificações:

        1. **Tipo de Pesquisa:** Satisfação
        2. **Empresa:** .. (se aplicável)
        3. **Quantidade de Perguntas:** 5
        4. **Instruções para o Usuário:** ...

        A pesquisa deve incluir perguntas dos seguintes tipos: "Texto Livre", "Escala de 1 a 10", "Múltipla Escolha" ou "Múltipla Seleção".

        As perguntas devem ser geradas no seguinte formato JSON, de acordo com o modelo exemplificado abaixo (sem sintaxe Markdown, apenas JSON puro):

        Exemplo de estrutura JSON:

        [
            {{
                "tipoPerguntaEnum": "TEXTO_LIVRE",
                "perguntaString": "Texto da pergunta de Texto Livre",
                "posicao": 0,
                "ocultarRelatorio": false
            }},
            {{
                "tipoPerguntaEnum": "ESCALA_1_10",
                "perguntaString": "Texto da pergunta de Escala 1 a 10",
                "posicao": 1,
                "ocultarRelatorio": false
            }},
            {{
                "tipoPerguntaEnum": "MULTIPLA_ESCOLHA",
                "perguntaString": "Texto da pergunta de Múltipla Escolha",
                "posicao": 2,
                "ocultarRelatorio": false,
                "opcoes": [
                    {{"opcaoString": "Opção 1", "posicao": 0, "peso": 1, "outros": false}},
                    {{"opcaoString": "Opção 2", "posicao": 1, "peso": 1, "outros": false}},
                    {{"opcaoString": "Opção 3", "posicao": 2, "peso": 1, "outros": false}},
                    {{"opcaoString": "Outros...", "posicao": 3, "peso": 1, "outros": true}}
                ]
            }},
            {{
                "tipoPerguntaEnum": "MULTIPLA_SELECAO",
                "perguntaString": "Texto da pergunta de Múltipla Seleção",
                "posicao": 3,
                "ocultarRelatorio": false,
                "opcoes": [
                    {{"opcaoString": "Opção 1", "posicao": 0, "peso": 1, "outros": false}},
                    {{"opcaoString": "Opção 2", "posicao": 1, "peso": 1, "outros": false}},
                    {{"opcaoString": "Opção 3", "posicao": 2, "peso": 1, "outros": false}},
                    {{"opcaoString": "Outros...", "posicao": 3, "peso": 1, "outros": true}}
                ]
            }}
        ]

        Instruções:
        1. O campo numérico "posicao" sempre começa no 0.
        2. O campo numérico "peso" sempre começa no 1.
        3. O campo booleano "outros" é obrigatório, mas geralmente tem valor "false". Quando "outros=true", essa opção deve ser a última.
        4. O campo "tipoPerguntaEnum" deve ser um dos seguintes: "TEXTO_LIVRE", "ESCALA_1_10", "MULTIPLA_SELECAO" ou "MULTIPLA_ESCOLHA".
        5. O campo "ocultarRelatorio" é um booleano que deve sempre ser "false".
        """
        
        # Apenas cria o objeto do prompt diretamente
        prompt = ChatPromptTemplate.from_template(template)
        
        # Retorna a chain para processar diretamente
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
                f.write(f"Assistente: {interaction['response']}\n")
                f.write(f"Tempo de Resposta: {interaction['time']:.2f} segundos\n\n")
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
            # Inicia o contador de tempo
            start_time = time.time()
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            response = qa_chain.invoke({
                "context": context,
                "question": question
            })
            
            # Calcula o tempo decorrido
            end_time = time.time()
            response_time = end_time - start_time
            
            # Adiciona ao histórico
            conversation_history.append({"question": question, "response": response, "time": response_time})
            
            # Exibe a resposta e o tempo de resposta
            print("\nAssistente:", response)
            print(f"(Tempo de resposta: {response_time:.2f} segundos)")
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
