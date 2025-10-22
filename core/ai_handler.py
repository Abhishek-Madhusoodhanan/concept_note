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



def find_internal_matches(preview, all_clarifications, internal_products):
    """
    Match client needs with internal products/modules
    Focus: Specific components, not full product dumps
    """
    products_content = ""
    for product in internal_products:
        pdf_text = extract_text_from_pdf(product.pdf_file)
        # Extract only relevant sections (first 2000 chars per product)
        products_content += f"\n\n=== {product.name} ===\n{pdf_text[:2000]}"
    
    prompt = f"""You are a solution architect. Match client needs with available products/modules.

CLIENT REQUIREMENTS:
{preview}

CLARIFICATIONS:
{all_clarifications}

AVAILABLE INTERNAL PRODUCTS/MODULES:
{products_content}

TASK: Recommend ONLY what's RELEVANT and USEFUL to the client.

CRITICAL RULES:
1. Recommend SPECIFIC MODULES/COMPONENTS, not entire products (unless truly needed)
2. Explain HOW each recommendation solves their problem
3. Be SELECTIVE - only include what adds real value
4. NO generic descriptions - be specific
5. If a product doesn't fit, DON'T force it

OUTPUT FORMAT (be concise):
[Module/Product Name]
Why: [One clear sentence explaining how it solves their problem]
Fit: [High/Medium - be honest]

Example:
Payment Gateway Module (from E-commerce Platform)
Why: Handles secure online payments with support for multiple providers
Fit: High

IMPORTANT: Only 3-5 most relevant recommendations. Quality over quantity."""

    response = model.generate_content(prompt)
    return response.text.strip()

def search_external_solutions(preview, all_clarifications):
    """
    Find external tools/modules/services (not competing products)
    Focus: Components that complement the solution
    """
    prompt = f"""You are a technology consultant. Suggest external tools/modules/services.

CLIENT REQUIREMENTS:
{preview}

CLARIFICATIONS:
{all_clarifications}

TASK: Recommend 5-7 SPECIFIC external tools/modules/services that fit their needs.

CRITICAL RULES:
1. Suggest TOOLS/MODULES/SERVICES, not competing full products
2. Focus on: frameworks, APIs, cloud services, libraries, platforms
3. Each recommendation must be ACTIONABLE and SPECIFIC
4. Explain WHY it fits (one clear sentence)
5. NO generic suggestions like "hire developers"
6. Prioritize: open-source, well-supported, modern options

OUTPUT FORMAT:
[Tool/Service Name]
Type: [Framework/API/Service/Platform]
Why: [How it helps their project]
Cost: [Free/Freemium/Paid]

Example:
Stripe API
Type: Payment Service
Why: Industry-standard payment processing with great documentation
Cost: Freemium (free to start, transaction fees apply)

Be practical and specific."""

    response = model.generate_content(prompt)
    return response.text.strip()

def generate_concept_note(description, highlight_points, document_content, client_vision, extracted_requirements, solution_design, external_features, implementation_plan, reference_context):
    """
    Generate a detailed, client-ready concept note with structured natural flow.
    """

    concept_prompt = f"""
Generate a comprehensive, personalized concept note with natural language flow.

PROJECT INPUTS:
Description: {description}
Highlights: {highlight_points}
Detailed Content: {document_content}

CLIENT'S SOLUTION VISION:
{client_vision}

REQUIREMENTS ANALYSIS:
{extracted_requirements}

COMPREHENSIVE SOLUTION DESIGN:
{solution_design}

EXTERNAL MARKET FEATURES:
{external_features}

IMPLEMENTATION PLAN:
{implementation_plan}

{reference_context}

CRITICAL INSTRUCTIONS FOR PROJECT TITLE:
1. Analyze the PROJECT INPUTS and DETAILED CONTENT to identify the ACTUAL project/client name
2. Look for mentions of: company names, project names, client organizations, product names
3. DO NOT use generic words like "description", "document", or field labels
4. If a clear client name exists (like "Convo AI", "XYZ School", "ABC Hospital"), USE IT
5. Create a professional title that reflects the REAL project essence

GENERATE CONCEPT NOTE WITH THIS STRUCTURE:

[Create Compelling Project Title - Based on project essence, 5-10 words]


1. EXECUTIVE SUMMARY

[Write 2-3 comprehensive paragraphs that capture the unique value proposition of this specific project. Include: the key business opportunity, competitive advantages of the proposed solution, and clear expectations for transformative impact. Make it compelling and specific to THIS project.]


2. PROBLEM STATEMENT

[Write 3-4 detailed paragraphs that articulate the specific business problem this project addresses. Include: current limitations and challenges, why solving this problem is strategically important, connection to broader market opportunities, and the case for urgent action. Base everything on the actual project details provided.]


3. CLIENT REQUIREMENTS & PROPOSED SOLUTIONS

Client Requirements:

[Analyze the DETAILED CONTENT section and organize the client's requirements into clear, logical categories. Present them as flowing text that comprehensively covers all needs mentioned in the preview. Do NOT just copy-paste the preview - synthesize it into organized requirement categories.]

Our Proposed Solution:

[Based on the SOLUTION DESIGN provided, describe the comprehensive solution architecture. Explain how the recommended internal products/modules address the requirements. Make it flow naturally as a solution narrative, not just a list. Show integration and how components work together.]


4. INNOVATIVE FEATURES & EXTERNAL ENHANCEMENTS

[Using the EXTERNAL MARKET FEATURES provided, describe additional capabilities that can enhance the solution. Present them as value-adds that ensure future scalability and competitive advantage. Make it specific to this project's domain.]


5. IMPLEMENTATION PLAN

[Create a realistic implementation timeline based on the project scope. Suggest phases like:
- Phase 1: Requirements finalization and design
- Phase 2: Core development and integration
- Phase 3: Testing, training, and deployment

Mention timeline considerations from the CLARIFICATIONS if any were provided about deadlines or timelines.]


6. EXPECTED BUSINESS IMPACT

[Write a comprehensive opening paragraph covering the quantifiable ROI and business benefits expected from this project. Include: operational efficiency improvements, cost savings, competitive advantages, both short-term (6-12 months) and long-term (2-3 years) value creation, and relevant KPIs. Make it specific to this project.]

Key Stakeholders and Their Benefits:

[Look at the TARGET USERS & STAKEHOLDERS section from the DETAILED CONTENT. For EACH stakeholder group mentioned there, write a detailed benefit description. Format as:

Stakeholder Group Name: Comprehensive description of benefits they'll experience, including how the solution addresses their pain points, improves workflows, enhances capabilities, and creates opportunities. Include both immediate and long-term benefits.

Continue for ALL stakeholder groups from the preview.]

[Write a closing paragraph about the comprehensive support strategy for smooth transition and adoption. Include mention of training and adoption frameworks tailored to each stakeholder group.]


7. CONCLUSION

[Write a powerful 3-4 paragraph conclusion that: summarizes the unique value proposition, reinforces competitive advantages, emphasizes strategic importance for the client's future, and presents a clear call-to-action with an inspiring vision statement. Make it compelling and specific to THIS project.]


CRITICAL REQUIREMENTS:
- Use the ACTUAL content from the sections provided - don't make up generic content
- Every section must be specific to this project
- NO instructional placeholders like "PARAGRAPH 1" or "[Insert content]"
- NO brackets in the final output
- Write in natural, flowing paragraphs
- Professional business document tone throughout
- Synthesize information, don't just copy-paste sections
- Ensure logical flow between all sections

Generate the complete personalized concept note now:
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