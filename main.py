import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tempfile
import spacy
from collections import Counter
import re

# Initialize AWS Bedrock client and embeddings
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def data_integration(uploaded_file):
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
    return vectorstore_faiss

def get_lamma3_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512, 'temperature': 0.5, 'top_p': 0.9})
    return llm

# Prompt template to analyze resume against JD and find skill gaps
prompt_template = """
You are an expert job match evaluator. Given the resume and job description, analyze the content and provide a percentage match based on how well the resume aligns with the job requirements.

**Job Description:**
{job_description}

**Resume:**
{context}

**Skill Gap Analysis:**
1. List the skills mentioned in the job description.
2. List the skills mentioned in the resume.
3. Identify any gaps between the job description's required skills and the resume's skills.

**Output:**
1. Percentage match between the resume and job description.
2. Summary of key strengths.
3. List of any significant gaps in the skills required.
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "job_description"]
)

def extract_skills(text):
    """Extract skills dynamically from the resume text using NLP and predefined skills."""
    # Predefined skills for quick matching
    predefined_skills = ["Python", "Java", "C++", "SQL", "AWS", "JavaScript", "HTML", "CSS", "React", "Node.js", "Docker", "Kubernetes"]
    
    # Extract skills using predefined list
    found_skills = [skill for skill in predefined_skills if re.search(r'\b' + skill + r'\b', text, re.IGNORECASE)]
    
    # Load the pre-trained spaCy model for entity recognition
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    # Extract skills using NER
    dynamic_skills = []
    for ent in doc.ents:
        if ent.label_ in ["SKILL", "ORG", "PRODUCT"]:
            dynamic_skills.append(ent.text)
    
    # Combine and deduplicate skills
    skills = list(set([skill.lower() for skill in found_skills + dynamic_skills]))
    return skills

def analyze_skill_gap(resume_text, jd_text):
    """Analyze the skill gap between the resume and job description."""
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    # Normalize skills for more accurate matching
    resume_skills_normalized = set(skill.lower() for skill in resume_skills)
    jd_skills_normalized = set(skill.lower() for skill in jd_skills)
    
    # Finding skill gaps
    missing_skills = list(jd_skills_normalized - resume_skills_normalized)

    return resume_skills, jd_skills, missing_skills

def get_skill_gap_analysis(resume_text, jd_text):
    """Generate a detailed skill gap analysis with percentages."""
    resume_skills, jd_skills, missing_skills = analyze_skill_gap(resume_text, jd_text)

    # Calculate match percentage
    matched_skills = list(set(resume_skills) & set(jd_skills))
    match_percentage = (len(matched_skills) / len(jd_skills)) * 100 if jd_skills else 0

    # Detailed analysis
    analysis = {
        "Matched Skills": matched_skills,
        "Missing Skills": missing_skills,
        "Match Percentage": match_percentage,
        # "Total Resume Skills": len(resume_skills),
        # "Total JD Skills": len(jd_skills)
    }

    return analysis

def get_response_llm(llm, resume_text, job_description):
    # Use the LLMChain directly with the custom prompt
    chain = LLMChain(llm=llm, prompt=PROMPT)
    answer = chain.run(context=resume_text, job_description=job_description)
    return answer

def main():
    st.set_page_config("Resume-JD Matcher")
    
    st.header("Upload Resume and Job Description for Matching with Skill Gap Analysis💁")

    # File upload for resume
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type="pdf")
    # File upload for JD
    uploaded_jd = st.file_uploader("Upload Job Description (PDF)", type="pdf")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if uploaded_resume and uploaded_jd:
            if st.button("Vectors Update"):
                with st.spinner("Processing..."):
                    resume_docs = data_integration(uploaded_resume)
                    jd_docs = data_integration(uploaded_jd)
                    get_vector_store(resume_docs + jd_docs)
                    st.success("Vector store updated with resume and JD")

    if st.button("Evaluate Match and Analyze Skill Gaps"):
        if uploaded_resume and uploaded_jd:
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_lamma3_llm()

                # Extract text for LLM evaluation
                resume_docs = data_integration(uploaded_resume)
                resume_text = " ".join([doc.page_content for doc in resume_docs])

                jd_docs = data_integration(uploaded_jd)
                job_description = " ".join([doc.page_content for doc in jd_docs])

                # Get the LLM response
                match_result = get_response_llm(llm, resume_text, job_description)
                st.write(match_result)

                # Detailed Skill Gap Analysis
                analysis = get_skill_gap_analysis(resume_text, job_description)
                
                st.subheader("Skill Gap Analysis")
                st.write(f"**Matched Skills:** {', '.join(analysis['Matched Skills'])}")
                st.write(f"**Missing Skills:** {', '.join(analysis['Missing Skills'])}")
                st.write(f"**Percentage Match:** {analysis['Match Percentage']:.2f}%")
                # st.write(f"**Total Resume Skills:** {analysis['Total Resume Skills']}")
                # st.write(f"**Total JD Skills:** {analysis['Total JD Skills']}")
                
                # Recommendations based on match percentage
                if analysis["Match Percentage"] > 75:
                    st.write("**Strengths:** The resume strongly aligns with the job description, covering most of the required skills.")
                elif 50 <= analysis["Match Percentage"] <= 75:
                    st.write("**Strengths:** The resume has a moderate alignment with the job description. Some skills match, but there are also notable gaps.")
                else:
                    st.write("**Strengths:** The resume has a weak alignment with the job description. Many key skills required by the job are missing.")
                
                st.write("**Additional Recommendations:** Consider improving the resume by acquiring or highlighting experience in the missing skills to better match the job description.")
                
                st.success("Done")
        else:
            st.warning("Please upload both a resume and a job description.")

if __name__ == "__main__":
    main()
