"""
System Instructions and Prompt Templates for Resume Analysis.

Defines the prompts used for grounded Q&A and specific Resume Intelligence features.
"""

RAG_SYSTEM_INSTRUCTION = (
    "You are a Senior AI recruiter and technical resume analyst.\n"
    "Your objective is to answer questions about a candidate's resume using ONLY the provided retrieved context.\n\n"
    "CRITICAL GROUNDING RULES:\n"
    "1. Ground your answers strictly in the retrieved text. Do NOT extrapolate or assume.\n"
    "2. If the answer cannot be found or inferred from the provided context, you MUST state exactly:\n"
    "   'The requested information is not available in the resume.'\n"
    "3. Do NOT make up any certifications, dates, skills, or job duties.\n"
    "4. Do NOT use external training data knowledge to invent background details about the candidate."
)

# Standard template for RAG Q&A
RAG_PROMPT_TEMPLATE = """
Context from the candidate's resume:
---------------------------------------------
{context}
---------------------------------------------

User Question: {question}

Provide a concise, professional, and well-structured answer:
"""

# Prompt templates for Resume Intelligence features
SUMMARY_PROMPT = """
Based on the candidate's resume context, provide a professional summary (2-3 paragraphs).
Focus on their core expertise, years of experience, key technical achievements, and career path.
"""

SKILLS_PROMPT = """
Extract all technical skills mentioned in the resume. 
Group them logically into categories (e.g., Programming Languages, Frameworks/Libraries, Databases, Tools/Devops, AI/Machine Learning).
Provide the output as a clean, bulleted list.
"""

PROJECTS_PROMPT = """
Extract all projects mentioned in the candidate's resume.
For each project, list:
- Project Name (if available)
- Role / Contributions
- Summary of what was built and its business impact
- Technologies used in the project
Format as a clean, structured report.
"""

EDUCATION_PROMPT = """
Extract the candidate's educational history.
For each institution, list:
- Institution Name
- Degree and Major
- Graduation Year or Date range
- Honors/GPA (if mentioned)
"""

EXPERIENCE_PROMPT = """
Extract the candidate's work experience.
For each job, list:
- Job Title
- Company Name
- Duration (Start and End dates)
- Key Responsibilities and Impact (bullet points)
"""

CERTIFICATIONS_PROMPT = """
Extract all certifications, licenses, or professional training accomplishments mentioned.
List the name of the certification, the issuing body (if available), and the year obtained.
If no certifications are mentioned, state that none are found.
"""

STRENGTHS_PROMPT = """
Based on the resume context, analyze the candidate's profile and list their top 3-5 strengths.
Explain why each strength is significant and cite supporting details from their experience.
"""

WEAKNESSES_PROMPT = """
Based on the resume context, perform a critical review of the candidate's profile.
Identify 2-3 potential gaps, weaknesses, or areas of improvement (e.g., missing standard skills for their level, short tenures, lack of certifications, or lack of quantitative metrics).
Provide constructive explanations.
"""

INTERVIEW_QUESTIONS_PROMPT = """
Based on the candidate's resume, generate 5-7 tailored technical and behavioral interview questions.
Each question should specifically target an experience, project, or technology mentioned in their profile.
Provide brief guidelines for what a strong answer should cover.
"""

MATCH_PROMPT_TEMPLATE = """
You are matching a candidate's resume against a target Job Description.

Target Job Description:
---------------------------------------------
{job_description}
---------------------------------------------

Retrieved Resume Context:
---------------------------------------------
{context}
---------------------------------------------

Analyze the match and provide a response in the following format:

### MATCH SCORE
[Provide a percentage score from 0% to 100% representing how well the candidate's skills and experience align with the job description.]

### MATCH EXPLANATION
- **Key Alignments**: [Bullet points listing skills, experience, and background details that match the requirements.]
- **Key Gaps**: [Bullet points listing required skills, experience, or attributes that are missing or weak in the candidate's profile.]
- **Recommendation**: [A 2-3 sentence final verdict on whether the candidate should be interviewed.]
"""
