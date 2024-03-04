import os
from dotenv import load_dotenv
from langchain.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.chains import RetrievalQA
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import DeepLake
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.text_splitter import RecursiveCharacterTextSplitter



load_dotenv()


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(model="../models/ggml-model-q4_0.bin", 
              callback_manager=callback_manager, verbose=True)
embeddings = GPT4AllEmbeddings(model="../models/ggml-model-q4_0.bin")

my_activeloop_org_id = os.getenv("ACTIVE_LOOP_ORG_ID")
my_activeloop_dataset_name = os.getenv("ACTIVE_LOOP_DATASET")

dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)


# create new documents
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset
db.add_documents(docs)



retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm,
	chain_type="stuff",
	retriever=db.as_retriever()
)


tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]
agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

response = agent.run("When was Napoleone born?")
print(response)