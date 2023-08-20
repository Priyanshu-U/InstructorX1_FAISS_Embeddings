from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM, Conversation
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings

from PyPDF2 import PdfReader


def get_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def create_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks


def vector_store(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vectors_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectors_store


def main():
    model_id = 'facebook/blenderbot-400M-distill'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=False)

    converse = pipeline(
        "conversational",
        model=model,
        tokenizer=tokenizer,
        max_length=100
    )
    pdf = open('Manuals/Priyanshu_Upadhyay_Resume.pdf', mode='rb')
    text = get_text(pdf)
    chunks = create_chunks(text)

    Vector_FAISS = vector_store(chunks)

    print(Vector_FAISS)

    conversation1 = Conversation("Hi How are you?")
    print(converse([conversation1]))


if __name__ == '__main__' :
    main()