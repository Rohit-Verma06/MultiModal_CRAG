from langgraph.graph import StateGraph , START , END
from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict , List , Literal , Annotated
from langchain_core.documents import Document
from pydantic import Field , BaseModel
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langgraph.graph.message import add_messages 
from langchain_core.messages import BaseMessage , HumanMessage
import re
import base64 
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.2)
image_model = ChatOpenAI(model = "gpt-4o-mini"  , temperature = 0)
GLOBAL_VECTOR_STORE = None
STORE_FILENAMES = set()
# GLOBAL_RETRIEVER = None

class cragstate(TypedDict):
    query : str 

    messages : Annotated[List[BaseMessage] , add_messages]

    specific_file : str 

    route_node : Literal["Retriever" , "Direct_Chat" , "Research"]

    rel_docs : List[Document] =Field(description="Documents relevant to the query")
    correct_docs : List[Document] = []
    ambigous_docs : List[Document] = []
    incorrect_docs : List[Document] = [] 
    research_docs : List[Document] = []
    refined_list_sentences : list = []
    ans : str 

def process_pdf(filepath : str , filename : str):
    global GLOBAL_VECTOR_STORE , STORE_FILENAMES
    if(filename in STORE_FILENAMES):
        return 
    
    loader = PyMuPDFLoader(filepath)
    docs = loader.load()
    for doc in docs:
        doc.metadata = {"source" : filename}
    splitter = RecursiveCharacterTextSplitter(separators = ["\n\n" , "\n" , " " , ""] , chunk_size = 1000, chunk_overlap = 200)
    chunks = splitter.split_documents(docs)
    if(not GLOBAL_VECTOR_STORE):
        GLOBAL_VECTOR_STORE = FAISS.from_documents(
            embedding = OpenAIEmbeddings(),
            documents = chunks
        )
    else:
        GLOBAL_VECTOR_STORE.add_documents(chunks)

    # GLOBAL_RETRIEVER = GLOBAL_VECTOR_STORE.as_retriever(search_type = "similarity" , search_kwargs = {"k" : 5})
    STORE_FILENAMES.add(filename)
    return 

def process_image(filepath : str , filename : str):
    global GLOBAL_VECTOR_STORE , STORE_FILENAMES
    if(filename in STORE_FILENAMES):
        return 
    
    with open(filepath , "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    prompt = (
        f"Extract all the text from this image perfectly. If there are tables, format them clearly using Markdown."
        f"If there are diagrams,  charts, or UI elements, describe them in detail."
        f"This content is from the file: {filename}"
    )
    extension = Path(filename).suffix.lower().lstrip('.')   # lstrip('.') used for removing left/start/leading '.'
    clean_extension = "jpeg" if extension == "jpg" else extension
    message = HumanMessage(
        content = [
            {"type" : "text" , "text" : prompt},
            {"type" : "image_url" , "image_url" : {"url" : f"data:image/{clean_extension};base64,{encoded_string}"} } 
        ]
    )

    extracted_image_text = image_model.invoke([message]).content
    doc = [Document(page_content=extracted_image_text , metadata = {"source" : filename})]

    splitter = RecursiveCharacterTextSplitter(separators=["\n\n" , "\n" , " ", ""] ,chunk_size = 1000, chunk_overlap = 200)
    chunks = splitter.split_documents(doc)

    if(not GLOBAL_VECTOR_STORE):
        GLOBAL_VECTOR_STORE = FAISS.from_documents(
            embedding=OpenAIEmbeddings(),
            documents=chunks
        )
    else:
        GLOBAL_VECTOR_STORE.add_documents(chunks)
    # GLOBAL_RETRIEVER = GLOBAL_VECTOR_STORE.as_retriever(search_type = "similarity" , search_kwargs = {"k" : 5})
    STORE_FILENAMES.add(filename)
    return 
        

class route_schema(BaseModel):
    route_node : Literal["Retriever" , "Direct_Chat" , "Research"] = Field(description = "The destination node to route the query to.")
    specific_file : str = Field(description = "The specific file name mentioned or implied in the query. Output 'all' if no specific file is meant." )

route_model = model.with_structured_output(schema = route_schema)
def route_node(state : cragstate)->dict:
    global STORE_FILENAMES
    query = state["query"]
    history = state["messages"]
    recent_history = "\n".join([f"{msg.type} : {msg.content}" for msg in history[-5:]] if history else None)
    filenames = ", ".join(STORE_FILENAMES) if STORE_FILENAMES else None
    template = PromptTemplate(
        template="""<system_role>
        You are an elite Intent Classification Agent for a Multi-Document RAG system.
        </system_role>

        <uploaded_files>
        {uploaded_files}
        </uploaded_files>

        <chat_history>
        {chat_context}
        </chat_history>

        <instructions>
        1. Analyze the <user_query>.
        2. Determine if the answer is likely within the <uploaded_files> (Retriever), requires live internet (Research), or is a general greeting/theory (Direct_Chat).
        3. Extract the exact filename from the list if the user targets one.
        </instructions>

        <examples>
        User Query: "How does the cooling system work in the PDF I just gave you?"
        Thought: User mentions 'the PDF', refers to uploaded content.
        Result: {{"route_node": "Retriever", "specific_file": "all"}}

        User Query: "What is the stock price of Nvidia right now?"
        Thought: Requires real-time data not in static PDFs.
        Result: {{"route_node": "Research", "specific_file": "all"}}
        </examples>

        <user_query>
        {query}
        </user_query>""",
        input_variables=["query", "uploaded_files", "chat_context"]
    )
    prompt = template.invoke({"query" : query , "uploaded_files" : filenames , "chat_context" : recent_history})
    output = route_model.invoke(prompt)
    state["specific_file"] = output.specific_file
    return {"route_node" : output.route_node}

def routing_condition(state : cragstate)->Literal["Retriever" , "Direct_Chat" , "Research"]:
    return state["route_node"]

def retriever_node(state : cragstate )->dict:
    query = state["query"]
    specific_file = state.get("specific_file" , "all")
    if(not GLOBAL_VECTOR_STORE):
        return {"rel_docs"  : []}
    if(specific_file.lower().strip() != "all"):
        rel_docs = GLOBAL_VECTOR_STORE.similarity_search(query ,k = 5,  filter = {"source" : specific_file})
    else:
        rel_docs = GLOBAL_VECTOR_STORE.similarity_search(query ,  k = 5)
    return {"rel_docs" : rel_docs}


class evaluator_schema(BaseModel):
    reasoning: str = Field(description="Step-by-step analysis of why this document matches or fails the query.")
    Confidence_Score : float = Field(description="Score the document's utility based on the Grading Scale provided in the prompt.")
evaluator_model = model.with_structured_output(schema = evaluator_schema)


def evaluator(state : cragstate)->dict:
    rel_docs = state.get("rel_docs" , [])
    query = state["query"]

    template = PromptTemplate(
        template = """<system>
You are an expert Retrieval Evaluator for a Corrective RAG (CRAG) system.
Your only job is to evaluate if a retrieved document contains useful information to answer the user's query.
</system>

<objective>
Analyze the document against the query. Explain your reasoning step-by-step, then assign a Confidence Score.
</objective>

<grading_scale>
- 0.8 to 1.0 (CORRECT): The document contains a direct answer, a core definition, OR specific examples that ground the answer.
- 0.3 to 0.79 (AMBIGUOUS): The document mentions relevant terms or concepts, but lacks actionable details to fully answer the query.
- 0.0 to 0.29 (INCORRECT): The document is entirely unrelated, consists only of metadata, or is actively unhelpful.
</grading_scale>

<special_rules>
- SUMMARY EXCEPTION: If the user query is broad (e.g., "summarize", "what is this report about"), ANY document containing actual file content is highly relevant. Score it >= 0.8.
- PARTIAL MATCH: If a document answers even a small part of a multi-part question, it is at least AMBIGUOUS (>= 0.3), NOT INCORRECT.
- IGNORE FORMATTING: Do not penalize documents for cut-off sentences or poor markdown parsing. Evaluate the underlying facts.
</special_rules>

<data_to_evaluate>
[User_Query]: {query}

[Document]: 
{document}
</data_to_evaluate>""",
        input_variables=["query", "document"]
    )

    evaluator_chain = template | evaluator_model
    batch_input = [{"query" : query , "document" : doc.page_content} for doc in rel_docs]
    output = evaluator_chain.batch(batch_input)

    correct_docs = []
    ambigous_docs = []
    incorrect_docs = []
    for document , score in zip(rel_docs , output):
        if(score.Confidence_Score<0.3):
            incorrect_docs.append(document)
        elif(score.Confidence_Score>=0.3  and score.Confidence_Score<0.8):
            ambigous_docs.append(document)
        else:
            correct_docs.append(document)
    
    return {"correct_docs" : correct_docs  , "ambigous_docs" : ambigous_docs , "incorrect_docs" : incorrect_docs}


def check_condition(state: cragstate)->Literal["Research" , "Refine"]:
    global GLOBAL_VECTOR_STORE
    if(not GLOBAL_VECTOR_STORE):
        return "Refine"
    if(not state.get("correct_docs" , []) and not state.get("ambigous_docs" , [])):
        return "Research"
    elif(not state.get("correct_docs" , []) and state.get("ambigous_docs" , [])):
        return "Research"
    elif(state.get("correct_docs" , []) and not state.get("ambigous_docs" , [])):
        return "Refine"
    else:
        return "Refine"


tool = TavilySearchResults(max_results = 5)
def tavily_search(query : str):
    output = tool.invoke({"query" : query})
    output_list = [Document(page_content = el.get("content" , "")) for el in output]
    return output_list



def research(state : cragstate)->dict:
    query = state["query"]
    history = state["messages"]
    recent_history = "\n".join([f"{msg.type} : {msg.content}" for msg in history[-5 : ]] if history else "No previous history") 
    template = PromptTemplate(
    template="""<task>
    Convert the user's conversational intent into a 'Search Cluster' of 3 distinct queries to ensure 100% coverage.
    </task>

    User Topic: {query}
    Context: {chat_context}

    Output format:
    1. [Broad Search]
    2. [Technical/Deep-Dive Search]
    3. [Recent/News Search]

    Queries:""",
        input_variables=["chat_context" , "query"]
    )
    output = model.invoke(template.invoke({"query" : query , "chat_context" : recent_history})).content.strip()
    web_results = tavily_search(output)
    return {"research_docs" : web_results}


class refine_schema(BaseModel):
    keep : bool 
refine_model = model.with_structured_output(schema = refine_schema)


def break_into_sentences(text : str)->List[str]:
    text = re.sub(r"\s+" , " " ,text).strip()
    sentences = re.split(r"(?<=[.!?])\s+" , text)
    return sentences

def refine(state : cragstate)->dict:
    query = state["query"]
    # correct_string = " ".join(doc.page_content for doc in state["correct_docs"]).strip()
    ambigous_string = " ".join(doc.page_content for doc in state.get("ambigous_docs" , [])).strip()
    research_string = " ".join(doc.page_content for doc in state.get("research_docs" , [])).strip()
    finalstring = ambigous_string + "\n" + research_string
    sentences = break_into_sentences(finalstring)
    batch_input = [{"query" : query , "sentence" : sentence} for sentence in sentences ]
    template = PromptTemplate(
        template = """You are a strict but helpful relevance filter.\n"
        "Your goal is to evaluate if a given sentence is useful for answering the specific question provided.\n\n"
        "RELEVANCE CRITERIA:\n"
        "1. Direct Answer: The sentence contains a partial or full answer.\n"
        "2. Context/Definition: The sentence defines a term or provides necessary background for the topic.\n"
        "3. Supporting Detail: The sentence provides examples, data, or evidence related to the question.\n\n"
        "Be generous: If the sentence has even a small chance of being useful, set keep=true.\n"
        "If the sentence is completely unrelated (e.g., page numbers, headers, or unrelated topics), set keep=false.\n"
        "Output ONLY JSON.\n
        Here is the query -> {query}\n
        Here is the Sentence -> {sentence}.""",
        input_variables=["query" , "sentence"]
    )
    refine_chain = template | refine_model
    output = refine_chain.batch(batch_input)
    refined_list = []
    for sentence , to_keep in zip(sentences , output):
        if(to_keep.keep):
            refined_list.append(sentence)
    return {"refined_list_sentences" : refined_list}


def generate(state : cragstate)->dict:

    global GLOBAL_VECTOR_STORE
    if(not GLOBAL_VECTOR_STORE):
        ans = "I don't have any documents uploaded yet! Please upload a PDF so I can look that up for you."
        return {"ans" : ans , "messages" : [("ai", ans)]}
    
    history = state.get("messages" , [])

    if(len(history)>1):
        history = history[:-1]

    history = "\n".join([f"{el.type} : {el.content}" for el in history])
    query = state["query"]
    refined_sentences = state.get("refined_list_sentences" ,[])
    correct_docs = state.get("correct_docs" , [])
    correct_string = " ".join(doc.page_content for doc in correct_docs).strip()
    refined_string = " ".join(sentence for sentence in refined_sentences).strip()

    final_string = correct_string + "\n" + refined_string
    template = PromptTemplate(
    template="""<persona>
        You are a Senior Research Assistant. Your responses are objective, cited, and strictly grounded in the provided context.
        </persona>

        <context>
        {refined_string}
        </context>

        <rules>
        - If the <context> is insufficient, say "I cannot confirm this from your documents."
        - Never mention "According to the context" or "The documents state." Just provide the answer.
        - Every claim MUST be followed by a source reference if available.
        </rules>

        <user_query>
        {query}
        </user_query>""",
            input_variables=["query" , "refined_string" , "chat_history"]
    )
    ans = model.invoke(template.invoke({"query" : query , "refined_string" : final_string.strip() , "chat_history" : history}))
    return {"ans" : ans.content.strip() , "messages" : [ans]}

def Direct_Chat(state: cragstate)->dict:
    output = model.invoke(state["messages"])
    return {"messages" : [output]  , "ans" : output.content.strip()}

checkpoint = InMemorySaver()
graph = StateGraph(cragstate)
graph.add_node("Route_Node" , route_node)
graph.add_node("Retriever" , retriever_node)
graph.add_node("Evaluator" , evaluator)
graph.add_node("Research" , research)
graph.add_node("Refine" , refine)
graph.add_node("Generate" , generate)
graph.add_node("Direct_Chat" , Direct_Chat)
graph.add_edge(START , "Route_Node")
graph.add_conditional_edges("Route_Node" , routing_condition)
graph.add_edge("Retriever" , "Evaluator")
graph.add_conditional_edges("Evaluator" , check_condition)
graph.add_edge("Research" , "Refine")
graph.add_edge("Refine" , "Generate")
graph.add_edge("Generate" , END)
graph.add_edge("Direct_Chat" , END)
chatbot = graph.compile(checkpointer=checkpoint)

from fastapi import FastAPI, UploadFile, File
import tempfile
import os

app = FastAPI()

class ChatRequest(BaseModel):
    query : str = Field(description="The User's query")
    thread_id : str = Field(default = "session_1" , description="The thread_id of the current chat")

@app.post("/chat")
def chat(request : ChatRequest):
    query = request.query
    output = chatbot.invoke({"query" : query , "messages" : [("user" , query)]} , config = {"configurable" : {"thread_id" : request.thread_id}})
    return {"answer" : output.get("ans" , "Sorry I could not generate an answer")}

@app.post("/upload")
async def upload_documents(file : UploadFile = File(...)):
    filename = file.filename
    file_extension = Path(filename).suffix.lower() # .pdf
    clean_file_extension = "jpeg" if file_extension.lstrip('.') == "jpg" else file_extension.lstrip('.')
    with tempfile.NamedTemporaryFile(delete = False , suffix=file_extension) as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = tmp_file.name
    try:
        if(clean_file_extension == "pdf"):
            process_pdf(filepath = tmp_path , filename = filename)
        else:
            process_image(filepath = tmp_path , filename = filename)
        return {"message" : f"Successfully Processed {filename}"}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        
    
 


