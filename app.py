from fastapi import FastAPI, Form, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
import json
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from fastapi.staticfiles import StaticFiles
from langchain import PromptTemplate
from huggingface_hub import hf_hub_download

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

model_path = hf_hub_download(repo_id="MaziyarPanahi/BioMistral-7B-GGUF", filename="BioMistral-7B.Q4_K_M.gguf")

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.3,
    n_ctx=2048,
    top_p=1
)

print("LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

client = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = client.as_retriever()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response", response_class=HTMLResponse)
async def get_response(query: str = Form(...)):
     chain_type_kwargs = {"prompt": prompt}
     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
     response = qa(query)
     answer = response['result']
     source_document = response['source_documents'][0].page_content
     doc = response['source_documents'][0].metadata['source']
     response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
    
     res = Response(response_data)
     return res
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
