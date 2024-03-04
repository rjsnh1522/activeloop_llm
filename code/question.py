from langchain.llms import GPT4All
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler



template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(model="../models/ggml-model-q4_0.bin", 
              callback_manager=callback_manager, verbose=True)

# The Chains

llm_chain = LLMChain(prompt=prompt, llm=llm)

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
text = "Suggest 1 names for a company that makes eco friendly water bottles?"
print(llm_chain(text))

