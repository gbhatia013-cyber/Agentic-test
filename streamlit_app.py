
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser

# Set up the page
st.set_page_config(page_title="Resume Screening App", layout='wide')
st.title("Resume Screening with Gemini and LangChain")

# Input fields
job_requirements = st.text_area("Paste Job Requirements", height=200)
resume_text = st.text_area("Paste Resume Text", height=200)

# Run the screening
if st.button("Evaluate Resume"):
    if not job_requirements or not resume_text:
        st.warning("Please provide both job requirements and resume text.")
    else:
        # Set up Gemini LLM
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

        # Prompt template
        prompt_template = PromptTemplate.from_template(
            "Compare the following job requirements and resume.\n"
            "Job Requirements:\n{job_requirements}\n\n"
            "Resume:\n{resume_text}\n\n"
            "How well does the resume match the job requirements?"
        )

        # Build the chain
        chain = (
            RunnableMap({
                "job_requirements": lambda X: X["job_requirements"],
                "resume_text": lambda X: X["resume_text"]
            })
            | prompt_template
            | llm
            | StrOutputParser()
        )

        # Run the chain
        result = chain.invoke({
            "job_requirements": job_requirements,
            "resume_text": resume_text
        })

        # Display result
        st.subheader("Evaluation Result")
        st.write(result)
