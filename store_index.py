import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings

load_dotenv()

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# This is the "magic" line that reads your .env file
#load_dotenv() 
load_dotenv("/Users/poonamkumari/medical-chat-bot/.env")


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


extracted_data = load_pdf_files(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
