from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from openai import AzureOpenAI
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)

load_dotenv()

# Now get the keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# Safety Check: If the keys are missing, stop here with a helpful message
if PINECONE_API_KEY is None:
    raise ValueError("Could not find API keys. Check your .env file!")

# You don't actually need to set os.environ again if you used load_dotenv(),
# but if your library specifically requires it, this will now work:
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # deployment name, not model name
    messages=[{"role": "user", "content": "Hello"}]
)


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_embeddings()

index_name = "medical-chatbot"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
