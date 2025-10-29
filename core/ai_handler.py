import google.generativeai as genai
from django.conf import settings
import PyPDF2
import io
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

# Configure Gemini
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

def process_audio_with_gemini(audio_file):
    """
    Transcribe audio using Gemini API
    Supports: MP3, WAV, M4A, FLAC, AAC, OGG
    """
    try:
        import tempfile
        import os
        
        # Save uploaded file temporarily
        file_extension = os.path.splitext(audio_file.name)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        
        for chunk in audio_file.chunks():
            temp_file.write(chunk)
        temp_file.close()
        
        # Upload audio file to Gemini
        audio_file_gemini = genai.upload_file(temp_file.name)
        
        # Transcribe using Gemini
        prompt = """Transcribe this audio accurately. 
        Convert speech to text exactly as spoken.
        Include all details mentioned.
        Format the output as clean, readable text."""
        
        response = model.generate_content([prompt, audio_file_gemini])
        
        # Clean up
        os.unlink(temp_file.name)
        
        return response.text.strip()
        
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"
    

def generate_pre_preview_questions(raw_input, pdf_text=None, highlight_points=None):
    """
    Generate intelligent clarification questions BEFORE creating the preview.
    Questions are 100% based on the actual client input.
    Returns a structured list of questions with detected values for confirmation.
    """
    
    # Combine all available input
    combined_input = f"""
USER'S DESCRIPTION:
{raw_input}

HIGHLIGHT POINTS:
{highlight_points if highlight_points else "None provided"}

PDF CONTENT:
{pdf_text[:3000] if pdf_text else "No PDF uploaded"}
"""
    
    prompt = f"""You are a business analyst reviewing initial project information. Your task is to identify missing or unclear information and generate smart clarification questions.

AVAILABLE INFORMATION:
{combined_input}

TASK: Analyze the input and generate 3-5 intelligent clarification questions. Each question should:
1. Be specific to what's actually mentioned (or missing) in the input
2. Help improve the quality of the concept note
3. Include detected values if found (for confirmation)
4. Be optional (client can skip if not relevant)

QUESTION CATEGORIES (only ask if relevant):
- Client/Organization Identification
- Project Budget/Investment Range
- Timeline/Deadline Expectations
- Scale/User Volume
- Existing Systems/Integration Needs
- Supporting Documentation
- Specific Technical Requirements
- Key Stakeholders/Decision Makers

RESPONSE FORMAT (JSON):
[
  {{
    "id": 1,
    "category": "client_identification",
    "question": "Is '[Detected Name]' the official client/organization name?",
    "detected_value": "[Extracted Name]",
    "field_type": "confirmation",
    "importance": "critical",
    "skip_allowed": false
  }},
  {{
    "id": 2,
    "category": "supporting_docs",
    "question": "Do you have any supporting documents (RFP, technical specs, wireframes) that would help us understand the requirements better?",
    "detected_value": null,
    "field_type": "yes_no_upload",
    "importance": "medium",
    "skip_allowed": true
  }},
  {{
    "id": 3,
    "category": "budget",
    "question": "What is your estimated budget or investment range for this project?",
    "detected_value": null,
    "field_type": "text_input",
    "importance": "high",
    "skip_allowed": true
  }}
]

CRITICAL RULES:
- Only generate questions for information that's truly unclear or missing
- If client name is clearly stated, ask for confirmation
- Always ask about supporting documents unless multiple PDFs already uploaded
- Don't ask generic questions - be specific to their project
- Maximum 5 questions - prioritize the most important
- Set skip_allowed=false only for absolutely critical information
- field_type options: "confirmation", "text_input", "yes_no", "yes_no_upload", "textarea"
- importance levels: "critical", "high", "medium", "low"

Return ONLY the JSON array, no additional text.
"""

    try:
        response = model.generate_content(prompt)
        questions_text = response.text.strip()
        
        # Clean up markdown code blocks if present
        if questions_text.startswith('```'):
            questions_text = questions_text.split('```')[1]
            if questions_text.startswith('json'):
                questions_text = questions_text[4:]
        
        questions = json.loads(questions_text)
        return questions
    
    except Exception as e:
        print(f"Error generating pre-preview questions: {e}")
        # Fallback to basic questions
        return [
            {
                "id": 1,
                "category": "client_identification",
                "question": "Please confirm or provide the client/organization name for this project",
                "detected_value": None,
                "field_type": "text_input",
                "importance": "critical",
                "skip_allowed": False
            },
            {
                "id": 2,
                "category": "supporting_docs",
                "question": "Do you have any supporting documents (RFP, specifications, wireframes) to share?",
                "detected_value": None,
                "field_type": "yes_no_upload",
                "importance": "medium",
                "skip_allowed": True
            }
        ]

def generate_preview(raw_input, highlight_points):
    """
    Convert raw client input into a DETAILED, BEAUTIFULLY FORMATTED preview
    """
    prompt = f"""You are a senior business analyst. Transform client requirements into a comprehensive, professional document.

CLIENT'S RAW INPUT:
{raw_input}

KEY POINTS TO EMPHASIZE:
{highlight_points if highlight_points else "None specified"}

CRITICAL INSTRUCTION: Analyze the CLIENT'S ACTUAL INPUT and create a preview SPECIFIC to their project. DO NOT use generic school/hospital/e-commerce examples unless that's what the client described.

TASK: Create a detailed preview that looks like a professional business document.

FORMAT STRUCTURE:

PROJECT TITLE & OVERVIEW
────────────────────────────────────────

[Extract or create a clear project title from the client's input]

[Write 2-3 comprehensive paragraphs explaining THIS SPECIFIC project based on what the client described. Each paragraph should be 4-6 sentences. Make it flow naturally. Use confident language but stay true to the client's vision.]


PRIMARY OBJECTIVES
────────────────────────────────────────

[Identify 5-8 specific objectives from the client's input. Write each as a clear statement.]

- [Objective 1 based on client's actual needs]
- [Objective 2 with measurable outcomes if mentioned]
- [Objective 3 focusing on user benefits from their description]
- [Continue with 5-8 objectives total - all must relate to THIS project]


TARGET USERS & STAKEHOLDERS
────────────────────────────────────────

[Identify who will use this system based on the client's input. If they mentioned teachers/students, use those. If they mentioned customers/admins, use those. If they mentioned doctors/patients, use those.]

[Stakeholder Group 1 from client input]: [2-3 sentences describing their needs, challenges, and how this helps them based on the project description]

[Stakeholder Group 2 from client input]: [2-3 sentences describing their needs and benefits specific to this project]

[Stakeholder Group 3 from client input]: [2-3 sentences describing their role and benefits]

[Continue for all stakeholder groups mentioned or implied in the client's input]


CORE FUNCTIONAL REQUIREMENTS
────────────────────────────────────────

[Identify 5-7 major functional areas from the CLIENT'S description. Each should have a descriptive title and explanation.]

[Requirement 1 Title - extracted from client needs]
[Write 2-3 sentences explaining what this requirement does, why it matters, and how it serves the users. Base this entirely on the client's input, not generic examples.]

[Requirement 2 Title - extracted from client needs]
[Write 2-3 sentences specific to this project's needs.]

[Requirement 3 Title - extracted from client needs]
[Write 2-3 sentences specific to this project's needs.]

[Continue with all major requirements identified from the input]


SPECIAL REQUIREMENTS & UNIQUE FEATURES
────────────────────────────────────────

[Look for unique aspects in the client's input - AI, automation, specific integrations, special workflows, etc.]

[Feature Category 1 if mentioned]: [2-3 sentences about this specific capability for this project]

[Feature Category 2 if mentioned]: [2-3 sentences about how this works in their context]

[Feature Category 3 if mentioned]: [2-3 sentences about implementation approach]


TECHNICAL CONSIDERATIONS
────────────────────────────────────────

[Identify 5-8 technical requirements based on the project scope and industry. Consider: scale, security, integrations, platforms, compliance]

- [Technical requirement 1 relevant to this project]
- [Technical requirement 2 based on industry/domain]
- [Technical requirement 3 addressing scalability/security]
- [Continue with technical needs specific to this solution]


EXPECTED OUTCOMES & BENEFITS
────────────────────────────────────────

[Identify 5-7 specific, measurable benefits this project will deliver based on the objectives and requirements]

- [Benefit 1 with measurable outcome if possible]
- [Benefit 2 specific to the stakeholders mentioned]
- [Benefit 3 addressing business value]
- [Continue with 5-7 benefits total]


CRITICAL FORMATTING RULES:
✓ Use section headers in CAPS followed by line separator (────)
✓ Write in paragraphs for descriptions (NOT bullet points unless listing)
✓ Each requirement gets 2-3 flowing sentences
✓ Professional business document tone
✓ Clear spacing between sections (double line break)
✓ 500-800 words total
✓ NO asterisks, NO markdown - just clean text with line separators

MOST IMPORTANT: Every section must be filled with content SPECIFIC to this client's input. Do NOT copy the example content about schools, students, or teachers unless the client specifically mentioned education. Analyze what THEY want and describe THEIR project.

OUTPUT: Professional document following the structure above with content specific to: {raw_input[:100]}"""

    response = model.generate_content(prompt)
    return response.text.strip()

def generate_clarification_questions(preview, conversation_history, raw_input):
    """
    Generate ONE intelligent clarification question at a time.
    - Asks for client name first if not found.
    - Only asks if something important is missing.
    - Avoids repeating or generic questions.
    - Stops after 4 total clarifications or when all key info is clear.
    """

    # --- Combine conversation so far ---
    asked_and_answered = ""
    questions_asked = []
    for item in conversation_history:
        if "question" in item:
            questions_asked.append(item["question"].lower())
        if "question" in item and "answer" in item:
            asked_and_answered += f"Q: {item['question']}\nA: {item['answer']}\n\n"

    questions_asked_count = len(questions_asked)
    if questions_asked_count >= 4:
        return "NO_MORE_QUESTIONS"

    # --- Combine all known text to detect what’s missing ---
    text_combined = (raw_input or "") + "\n" + (preview or "") + "\n" + asked_and_answered.lower()

    # --- Step 1: Check if client name is missing ---
    if not any(word in text_combined.lower() for word in [
        "client", "organization", "company", "school", "university",
        "hospital", "ngo", "startup", "institute", "college"
    ]) and not any("client" in q for q in questions_asked):
        return "What is the name of the client, organization, or company this project is for?"

    # --- Step 2: Identify missing essential info ---
    missing_info = []
    if not any(keyword in text_combined.lower() for keyword in ["budget", "cost", "price", "funding", "investment"]):
        missing_info.append("budget")
    if not any(keyword in text_combined.lower() for keyword in ["timeline", "deadline", "duration", "timeframe", "month", "week"]):
        missing_info.append("timeline")
    if not any(keyword in text_combined.lower() for keyword in ["users", "students", "patients", "employees", "transactions", "scale"]):
        missing_info.append("scale")
    if not any(keyword in text_combined.lower() for keyword in ["integrate", "integration", "api", "system", "platform", "crm", "erp"]):
        missing_info.append("integration")

    # --- Step 3: If all key info is clear, stop asking ---
    if not missing_info:
        return "NO_MORE_QUESTIONS"

    # --- Step 4: Prioritize missing items logically ---
    priority_order = ["budget", "timeline", "scale", "integration"]
    for key in priority_order:
        if key in missing_info and not any(key in q for q in questions_asked):
            if key == "budget":
                return "What is your estimated budget or funding range for this project?"
            elif key == "timeline":
                return "When do you need this project to be delivered or operational?"
            elif key == "scale":
                return "Roughly how many users, patients, or people will use this system initially?"
            elif key == "integration":
                return "Do you have any existing systems or platforms this solution should integrate with?"

    # --- Step 5: Fallback – use LLM for fine-grained intelligence ---
    prompt = f"""You are a business analyst reviewing project details.

ORIGINAL INPUT:
{raw_input}

PREVIEW SUMMARY:
{preview}

ALREADY CLARIFIED:
{asked_and_answered if asked_and_answered else "Nothing yet."}

TASK:
If something essential to build a concept note is unclear, ask ONE very specific question.
Otherwise, return exactly: "NO_MORE_QUESTIONS".

Essential items: client name, budget, timeline, scale, integration needs.

Be smart — only ask if it’s truly required for concept note accuracy.
Avoid repeating already asked or answered questions.
Return just the single question text or "NO_MORE_QUESTIONS".
"""

    response = model.generate_content(prompt)
    return response.text.strip()


# ai_handler.py

def find_internal_matches(preview, all_clarifications, internal_products):
    """
    Extract relevant FEATURES from internal products that can be integrated into client's project
    Returns: Bullet-point recommendations explaining specific features/capabilities that fit their needs
    """
    
    # Step 1: Extract keywords for smart filtering
    keywords_prompt = f"""Extract 5-7 key technology/domain keywords from this project.
    Return ONLY comma-separated keywords (e.g., "healthcare, automation, AI, mobile, payment").
    
    PROJECT: {preview[:1000]}
    CLARIFICATIONS: {all_clarifications[:500]}"""
    
    try:
        keywords_response = model.generate_content(keywords_prompt)
        keywords = keywords_response.text.strip().lower()
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        keywords = ""
    
    # Step 2: Smart product filtering and feature extraction
    products_content = ""
    relevant_products = []
    keyword_list = [k.strip() for k in keywords.split(',')] if keywords else []
    
    for product in internal_products:
        product_name = product.name.lower()
        product_desc = (product.description or "").lower()
        
        # Calculate relevance score
        relevance_score = 0
        for keyword in keyword_list:
            if keyword in product_name:
                relevance_score += 3
            if keyword in product_desc:
                relevance_score += 2
        
        # Use cached extracted_text if available
        if hasattr(product, 'extracted_text') and product.extracted_text:
            pdf_text = product.extracted_text[:2500]
        else:
            try:
                pdf_text = extract_text_from_pdf(product.pdf_file)[:2500]
            except Exception as e:
                print(f"Error reading {product.name}: {e}")
                continue
        
        # Include high-relevance products
        if relevance_score > 0 or len(relevant_products) < 5:
            char_limit = 3000 if relevance_score > 2 else 1500
            products_content += f"\n\n=== {product.name} ===\n"
            if product.description:
                products_content += f"Description: {product.description}\n"
            products_content += f"Available Features: {pdf_text[:char_limit]}\n"
            relevant_products.append((product.name, relevance_score))
    
    # Sort by relevance
    relevant_products.sort(key=lambda x: x[1], reverse=True)
    top_products = [p[0] for p in relevant_products[:8]]
    
    # Step 3: Generate feature-focused recommendations (IN BULLET POINTS)
    prompt = f"""Analyze which SPECIFIC FEATURES from our products can be ADDED to the client's project.

CLIENT NEEDS:
{preview[:800]}

KEY DETAILS:
{all_clarifications[:600]}

OUR PRODUCTS & FEATURES:
{products_content[:3000]}

FOCUS ON: {', '.join(top_products[:5])}

INSTRUCTIONS:
Write the output in BULLET POINTS format (• or -).
Each point should clearly describe:
- The feature name and source product
- How it integrates into the client's project
- The value or enhancement it brings

CRITICAL:
- Mention specific FEATURES, not entire products
- Use: "Integrate the [feature] from [product]"
- NOT: "Use [product] as the solution"
- Example: "• Integrate the AI diagnostic engine from Omnia to enable automated symptom analysis."
- Focus on modular, integrable components

Output 6–8 concise bullet points. After the bullet points, on a new line, ask the user for confirmation with the exact text:
---
Shall I proceed with these suggestions, or would you like to review/add other products?
"""

    # CORRECTED INDENTATION IS HERE
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating internal feature recommendations: {str(e)}"
def search_external_solutions(preview, all_clarifications):
    """
    Suggest external technologies (APIs, libraries, tools) that can be integrated
    Returns: Bullet-point recommendations explaining external features and capabilities
    """
    prompt = f"""Recommend SPECIFIC external technologies/features that can be integrated into this project.

CLIENT NEEDS:
{preview[:800]}

KEY DETAILS:
{all_clarifications[:600]}

INSTRUCTIONS:
Write the output as BULLET POINTS (• or -).
Each point should:
- Mention the specific tool, API, or library
- Explain briefly what it does
- Describe how it can be integrated and its value

CRITICAL:
- Suggest SPECIFIC features/APIs, not full platforms
- Example: "• Use Stripe Payment Intents API for secure, flexible online payment processing."
- NOT: "Use Shopify for e-commerce."
- Focus on 5–7 technologies maximum
- Cover: Cloud, APIs, Frontend, Backend, Database, AI/ML

Output 5–7 concise bullet points:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating external feature recommendations: {str(e)}"
def generate_concept_note(description, highlight_points, document_content, client_vision, extracted_requirements, solution_design, external_features, implementation_plan, reference_context):
    """
    Generate a polished, corporate-level concept note (2-3 pages) with a strategic, professional tone.
    Updated: Refined for MNC-level professionalism and removed all revenue-related content.
    """

    concept_prompt = f"""
Generate a comprehensive, professional concept note (STRICTLY 2-3 pages / 1200-1500 words) that reflects the tone and structure of a top-tier consulting or technology firm document. 
Use strategic, business-oriented language and focus on clarity, impact, and transformation — NOT on financials or revenue outcomes.

PROJECT INPUTS:
Description: {description}
Highlights: {highlight_points}
Detailed Content: {document_content}

CLIENT’S SOLUTION VISION:
{client_vision}

REQUIREMENTS ANALYSIS:
{extracted_requirements}

INTERNAL SOLUTIONS ANALYSIS (USE THIS EXACTLY AS PROVIDED):
{solution_design}

EXTERNAL TECHNOLOGIES ANALYSIS (USE THIS EXACTLY AS PROVIDED):
{external_features}

IMPLEMENTATION PLAN:
{implementation_plan}

{reference_context}

CRITICAL INSTRUCTIONS FOR PROJECT TITLE:
1. Derive the actual client or project name from the given inputs.
2. Identify company or organization names accurately.
3. Avoid generic placeholders like "document" or "description."
4. Use a format that highlights purpose and client (e.g., "AI-Enabled Health Consultation Platform for Precise Eye Hospital").
5. Ensure the title conveys value and professionalism.

CONTENT RULES:
❌ No repetition across sections.
❌ No overlapping narratives.
❌ No mention of revenue, profit, or monetary ROI.
✓ Maintain unique, relevant, and contextual information per section.
✓ Use confident, business-level English.
✓ Emphasize efficiency, innovation, and stakeholder impact.
✓ Keep paragraphs concise (3–5 sentences each).

GENERATE CONCEPT NOTE USING THIS STRUCTURE:

[Professional Project Title — Client-Focused and Descriptive]

1. ABOUT US
GAUDE BUSINESS AND INFRASTRUCTURE SOLUTIONS PVT LTD is a strategic business unit committed to delivering proactive value to clients through tailored, high-quality solutions. 
Operating from Technopark Campus, Trivandrum, under Kerala Start-Up Mission, GAUDE provides software development and managed services supported by a globally experienced management and technical team. 
The organization focuses on Application Development, Maintenance, and Managed Services, consistently ensuring optimized delivery, innovation, and quality that drive measurable impact for its clients.

2. EXECUTIVE SUMMARY
[Compose one strategic paragraph (100–120 words) summarizing:
- The transformative opportunity (1 sentence)
- The key client challenge (1 sentence)
- The proposed solution and differentiator (1 sentence)
- The expected organizational impact (1 sentence)
- The collaboration and value of partnership (1 sentence)
Use formal, results-oriented language.]

3. PROBLEM STATEMENT
[Write two concise paragraphs (180–220 words):
Paragraph 1 – Describe the client’s current state and operational challenges drawn from REQUIREMENTS and VISION.
Paragraph 2 – Highlight the strategic urgency and implications of inaction.
Avoid any mention of solutions or financial implications; focus only on contextual problems.]

4. PROPOSED SOLUTION
[Write 2–3 paragraphs (250–300 words) using INTERNAL SOLUTIONS ANALYSIS as the core:
Paragraph 1 – Overview of the proposed solution and technology landscape.
Paragraph 2 – Integration of GAUDE’s internal solutions with client needs (use provided paragraphs directly if available).
Paragraph 3 – Strategic advantage, differentiation, and alignment with client objectives.
Use forward-looking, outcome-driven language like “will enhance,” “will streamline,” “will empower.”]

5. KEY FEATURES AND FUNCTIONALITIES
[Structured format (350–400 words):
Start with: “The proposed solution encompasses key functionalities tailored to address the client’s operational and technical needs.”
List 6–8 essential, high-impact features relevant to primary, secondary, and administrative users.
Use concise feature descriptions (2 sentences each) explaining purpose and client relevance.
Integrate EXTERNAL TECHNOLOGIES and REQUIREMENTS content where appropriate.
❌ Do not include generic or financial features.]

6. IMPLEMENTATION APPROACH AND DEVELOPMENT PROCESS
[Write a structured, phase-wise roadmap (150–180 words):
“The implementation will follow a phased approach to ensure scalability and seamless adoption.”
Phase I – Foundation and Core Development
Phase II – Integration and Enhancement
Phase III – Optimization and Expansion
Each phase should specify scope, deliverables, and timeframes based on the IMPLEMENTATION PLAN.
Close with: “This phased execution ensures quality, adaptability, and stakeholder confidence.”]

7. EXPECTED OUTCOMES AND BENEFITS
[Two paragraphs (200–240 words):
Paragraph 1 – Focus on measurable operational outcomes such as efficiency gains, process optimization, scalability, and user experience improvement.
Paragraph 2 – Outline benefits for key stakeholder groups (end users, administrators, management) emphasizing collaboration, automation, and long-term sustainability.
❌ Exclude any reference to revenue, profit, or monetary value.]

8. CONCLUSION
[One concise paragraph (80–100 words):
Reiterate the project’s transformative value for the client.
Highlight GAUDE’s partnership-driven approach and commitment to innovation.
End with a confident, forward-looking statement outlining readiness for next steps and collaboration.]

FORMATTING REQUIREMENTS:
✓ Use section numbering (1., 2., 3., etc.)
✓ Section titles in **bold caps**
✓ Business-grade writing: precise, objective, and strategic
✓ Include client-specific terms where possible
✓ Avoid placeholders or brackets
✓ Keep total length 1200–1500 words (2–3 pages)
✓ Maintain brevity and professionalism throughout

TONE AND STYLE:
✓ Formal and authoritative
✓ Strategic, insightful, and client-centered
✓ Impact-driven and technically sound
✓ No marketing fluff — focus on clarity and execution
✓ Future-focused with “will” statements
✓ Never mention revenue or financial metrics

STRICT UNIQUENESS ACROSS SECTIONS:
1. About Us → Company profile only
2. Executive Summary → Overview and direction
3. Problem Statement → Challenges only
4. Proposed Solution → Solution architecture only
5. Key Features → Functional specifications only
6. Implementation → Execution roadmap only
7. Expected Outcomes → Operational and stakeholder benefits only
8. Conclusion → Vision alignment and next steps only

Generate the final polished concept note now.
"""

    response = model.generate_content(concept_prompt)
    return response.text.strip()


def generate_pdf(concept_note_text, client_name=None):
    import re
    from reportlab.lib import colors
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buffer = BytesIO()

    # ✅ Initialize cleaned_text from input
    cleaned_text = concept_note_text or ""  # fallback if None

    # 🔹 1. Remove "Concept Note - xxxx" lines and similar headers
    cleaned_text = re.sub(r"(?im)^\s*concept\s*note\s*[-:]\s*\w+\s*$", "", cleaned_text)

    # 🔹 2. Remove markdown and extra characters
    cleaned_text = cleaned_text.replace("**", "").replace("*", "")
    cleaned_text = cleaned_text.replace("═", "").replace("─", "").replace("_", "")
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)  # normalize extra spacing

    # 🔹 3. Convert markdown-style headers into formatted lines
    cleaned_text = re.sub(r"(?m)^##\s*", "", cleaned_text)  # remove ##
    cleaned_text = re.sub(r"(?m)^#\s*", "", cleaned_text)   # remove single #

    # 🔹 4. Split lines for rendering
    lines = cleaned_text.strip().split("\n")

    # Create PDF
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
    )

    styles = getSampleStyleSheet()

    client_style = ParagraphStyle(
        "ClientTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#4d9eff"),
        spaceAfter=14,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#4d9eff"),
        spaceAfter=10,
        spaceBefore=14,
        fontName="Helvetica-Bold",
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=11,
        leading=15,
        alignment=TA_LEFT,
        spaceAfter=8,
        textColor=colors.black,
    )

    story = []

    # Add client/project name at the top
    if client_name:
        story.append(Paragraph(client_name, client_style))
        story.append(Spacer(1, 0.3 * inch))

    # Parse content line-by-line
    for line in cleaned_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.1 * inch))
        elif line.isupper() and len(line) > 5:
            story.append(Paragraph(line, heading_style))
        else:
            story.append(Paragraph(line, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer
def extract_client_name_from_content(raw_input, formatted_preview, conversation_history):
    """
    Intelligently extract the actual client/project name from available data.
    Returns a clean, professional project title.
    """
    
    # Combine all available text
    all_text = f"{raw_input}\n{formatted_preview}\n"
    
    # Add conversation history
    for item in conversation_history:
        if "answer" in item:
            all_text += f"\n{item['answer']}"
    
    # First, try to find explicit client name from clarifications
    for item in conversation_history:
        question = item.get('question', '').lower()
        answer = item.get('answer', '').strip()
        
        if answer and ('client' in question or 'organization' in question or 'company' in question):
            # Found explicit client name answer
            return answer[:60]  # Use the client's direct answer
    
    prompt = f"""
Analyze this project information and extract the ACTUAL client/project name or create a descriptive title.

PROJECT INFORMATION:
{all_text[:3000]}

TASK: Identify the client/project name OR create a descriptive title based on what the project is about.

PRIORITY ORDER:
1. If you find a specific company/organization name → return it (e.g., "Convo AI", "ABC School")
2. If you find a project name → return it (e.g., "Student Management Portal")
3. If no specific name exists → create a clear descriptive title based on the domain and purpose

RULES:
- Look for: company names, organization names, project titles, product names
- If creating a descriptive title, base it on what the system DOES
- Keep it under 60 characters
- Make it professional and specific

GOOD examples:
- "Convo AI Voice Platform" (if Convo AI is mentioned)
- "AI-Powered Concept Note Generator" (descriptive based on function)
- "Healthcare Management System" (domain-based if no specific name)
- "Smart School Administration Platform" (domain + purpose)

BAD examples (avoid):
- "Description" ❌
- "System Development" ❌  
- "Project Overview" ❌
- Generic phrases that don't describe the actual project

Based on the content above, return ONLY the project/client name or descriptive title:
"""
    
    try:
        response = model.generate_content(prompt)
        extracted_name = response.text.strip()
        
        # Clean up the response
        extracted_name = extracted_name.replace('"', '').replace("'", "")
        
        # Remove common bad patterns
        bad_patterns = ['description', 'document', 'project overview', 'concept note', 'system development']
        extracted_lower = extracted_name.lower()
        
        for bad in bad_patterns:
            if bad == extracted_lower or (bad in extracted_lower and len(extracted_name) < 30):
                # If it's a bad generic name, create one based on domain
                if 'school' in all_text.lower() or 'education' in all_text.lower():
                    return "Education Management System"
                elif 'hospital' in all_text.lower() or 'health' in all_text.lower():
                    return "Healthcare Management Platform"
                elif 'voice' in all_text.lower() or 'call' in all_text.lower():
                    return "Voice Communication Platform"
                elif 'e-commerce' in all_text.lower() or 'shop' in all_text.lower():
                    return "E-Commerce Platform"
                else:
                    return "Business Solution Platform"
        
        return extracted_name[:60]  # Limit length
        
    except Exception as e:
        print(f"Error extracting client name: {e}")
        # Fallback: Try to infer from domain
        text_lower = all_text.lower()
        if 'school' in text_lower or 'education' in text_lower:
            return "Education Management System"
        elif 'hospital' in text_lower or 'health' in text_lower:
            return "Healthcare Management Platform"
        elif 'voice' in text_lower or 'ai' in text_lower:
            return "AI-Powered Solution Platform"
        else:
            return "Business Solution Platform"
        


def generate_ai_suggestion(selected_text, full_context, suggestion_type="improve"):
    """
    Generate AI suggestions for selected text within the preview
    
    Args:
        selected_text: The text the user selected
        full_context: The complete preview text for context
        suggestion_type: Type of suggestion ("improve", "expand", "simplify", "rephrase")
    
    Returns:
        Dict with original and suggested text
    """
    
    suggestion_prompts = {
        "improve": """Improve this text to make it more professional, clear, and impactful.
Keep the core meaning but enhance the language, structure, and clarity.
Make it concise yet comprehensive.""",
        
        "expand": """Expand this text with more details, examples, or explanations.
Add depth while maintaining professional tone and relevance to the project.""",
        
        "simplify": """Simplify this text to make it clearer and more straightforward.
Remove jargon, use simpler language, but keep the essential information.""",
        
        "rephrase": """Rephrase this text using different words and sentence structure.
Maintain the exact same meaning but present it in a fresh way.""",
        
        "alternative": """Generate an alternative version of this text.
Take a different angle or emphasis while covering the same points."""
    }
    
    prompt = f"""You are a professional business writer helping to refine a concept note.

FULL DOCUMENT CONTEXT:
{full_context[:2000]}

SELECTED TEXT TO IMPROVE:
"{selected_text}"

TASK: {suggestion_prompts.get(suggestion_type, suggestion_prompts["improve"])}

CRITICAL RULES:
- Keep the same length as the original (±20%)
- Maintain consistency with the document's tone
- Preserve all key facts and information
- Match the formatting style (bullet points stay as bullets, paragraphs as paragraphs)
- Do NOT add placeholder text or [brackets]
- Make it actionable and specific
- Ensure it flows naturally with surrounding text

OUTPUT FORMAT:
Return ONLY the improved text without any explanation, prefixes, or quotation marks.
Just the refined text that can directly replace the original.
"""

    try:
        response = model.generate_content(prompt)
        suggested_text = response.text.strip()
        
        # Clean up common AI artifacts
        suggested_text = suggested_text.strip('"\'`')
        if suggested_text.startswith('Here') or suggested_text.startswith('Sure'):
            # Remove explanatory prefixes
            lines = suggested_text.split('\n')
            suggested_text = '\n'.join(lines[1:]).strip()
        
        return {
            'success': True,
            'original': selected_text,
            'suggestion': suggested_text,
            'suggestion_type': suggestion_type
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error generating suggestion: {str(e)}",
            'original': selected_text
        }


def generate_multiple_suggestions(selected_text, full_context, count=3):
    """
    Generate multiple alternative suggestions for the selected text
    Useful for giving users options
    """
    suggestions = []
    
    suggestion_types = ["improve", "rephrase", "alternative"]
    
    for i, stype in enumerate(suggestion_types[:count]):
        result = generate_ai_suggestion(selected_text, full_context, stype)
        if result['success']:
            suggestions.append({
                'id': i + 1,
                'type': stype,
                'text': result['suggestion']
            })
    
    return suggestions