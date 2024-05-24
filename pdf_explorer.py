import os
import dotenv
from langchain_openai import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import streamlit as st

dotenv.load_dotenv('API.env')
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
print(OpenAI.api_key)

text_pdf_url = "https://www.medrxiv.org/content/10.1101/2021.07.15.21260605v1.full.pdf"

loader = PyPDFLoader(text_pdf_url)
file = loader.load_and_split()

prompt_text_template = """ Write a summary of the research paper for an artifical intelligence 
                        researcher that includes main points and any important details in 
                        bullet points.{text}
                       """
prompt_text = PromptTemplate(input_variables=["text"],
                       template = prompt_text_template,)

combine_prompt_template = """ you will be given the main points and any important details of the research 
                                paper in bullet points.Your goal is to give a final summary of the main 
                                research topic and the findings which will be useful to an artificial 
                                intelligence researcher to grasp what was done during the research work.
                                ```{text}``` FINAL SUMMARY =
"""

combine_prompt_text = PromptTemplate(input_variables=["text"],
                       template = combine_prompt_template,)

def pdf_summarizer(file_path, chunk_size, chunk_overlap, prompt_text):
    
    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0, openai_api_key = OpenAI.api_key)
    
    loader = PyPDFLoader(file_path)
    file = loader.load()
    
    file_text = [content.page_content for content in file]
    
    file_text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    file_chunks = file_text_splitter.create_documents(file_text)
    
    chain = load_summarize_chain(llm, chain_type = "stuff", prompt = prompt_text)
#     chain = load_summarize_chain(llm, chain_type = "map_reduce", map_prompt = prompt_text, combine_prompt = combine_prompt_text)
    summary = chain.invoke(file_chunks, return_only_outputs = True)
    
    return summary['output_text']

# print(pdf_summarizer(text_pdf_url, 1000, 20, prompt_text, combine_prompt_text))
print(pdf_summarizer(text_pdf_url, 1000, 20, prompt_text))

def main():
    st.set_page_config(page_title="PDF Explorer", page_icon=":book:", layout="wide")
    st.title("PDF Explorer")
    
    pdf_path = st.text_input("Enter the path to the PDF File:")
    if pdf_path != "":
        st.write("PDF file path was loaded successfully")
    
    user_prompt = st.text_input("Enter your prompt:") + """{text}"""
    prompt = PromptTemplate(input_variables=["text"],
                           template = user_prompt,)
    
    if st.button("Summarize"):
        summary = pdf_summarizer(pdf_path, 1000, 20, prompt)
        st.write(summary)
        
if __name__ == "__main__":
    main()


