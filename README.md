# 📄 Resume-JD Matcher with Skill Gap Analysis

Welcome to the Resume-JD Matcher! 🎉 This tool helps you evaluate how well a resume aligns with a job description by analyzing skills and identifying gaps. It leverages AWS Bedrock for embeddings, spaCy for NLP, and Streamlit for a user-friendly interface.

## 🛠️ Features

- **Upload PDF Files**: Supports uploading resumes and job descriptions in PDF format.
- **Vector Store Creation**: Creates and updates a local FAISS vector store with resume and job description data.
- **Skill Gap Analysis**: Analyzes and identifies skills gaps between the resume and job description.
- **LLM Integration**: Uses AWS Bedrock LLM to evaluate and generate a match percentage and analysis.
- **Visualization**: Displays results including matched skills, missing skills, and match percentage.

## 🚀 Getting Started

### Prerequisites

Ensure you have the following Python packages installed:

- `boto3`
- `streamlit`
- `langchain_community`
- `spacy`
- `faiss-cpu` (or `faiss-gpu` if you have a GPU)
- `tempfile`

You also need to download the `en_core_web_sm` model for spaCy:

```bash
python -m spacy download en_core_web_sm
Usage
Upload Resume and Job Description: Use the file uploaders to upload your PDF files.
Update Vector Store: Click the "Vectors Update" button to process and create/update the vector store.
Evaluate Match and Analyze Skill Gaps: Click the "Evaluate Match and Analyze Skill Gaps" button to get a detailed skill gap analysis and match percentage.
📈 Output
The tool will provide:

Matched Skills: Skills found in both the resume and job description.
Missing Skills: Skills required by the job description but missing in the resume.
Match Percentage: How well the resume matches the job description (in percentage).
Recommendations: Insights on strengths and areas for improvement.
💡 Tips
Ensure your PDF files are clean and well-formatted for better analysis.
Regularly update your vector store to reflect any changes in the resume or job description.
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing
Feel free to open issues or submit pull requests if you have suggestions or improvements!

📞 Contact
For any questions or feedback, please reach out to your-shubhamohdar26@gmail.com.
