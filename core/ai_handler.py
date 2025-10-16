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

TASK: Create a detailed preview that looks like a professional business document.

FORMAT EXACTLY LIKE THIS (copy this style):

PROJECT TITLE & OVERVIEW
────────────────────────────────────────

[Write project title on first line]

[Write 2-3 comprehensive paragraphs explaining the project. Each paragraph should be 4-6 sentences. Make it flow naturally like a professional document. Use clear, confident language.]


PRIMARY OBJECTIVES
────────────────────────────────────────

- Objective 1 written in clear, specific language
- Objective 2 with measurable outcomes where possible
- Objective 3 focusing on user benefits
- Continue with 5-8 well-defined objectives


TARGET USERS & STAKEHOLDERS
────────────────────────────────────────

Teachers: [2-3 sentences describing their needs, challenges, and how this helps them]

Students: [2-3 sentences describing their needs and benefits]

Parents: [2-3 sentences describing their role and benefits]

[Continue for all stakeholder groups]


CORE FUNCTIONAL REQUIREMENTS
────────────────────────────────────────

Student Information Management
The system will maintain comprehensive student profiles including academic records, attendance, health information, and learning preferences. This centralized database enables personalized learning experiences and data-driven decision making. Teachers and administrators can access relevant information instantly while maintaining data privacy compliance.

Lesson Planning & Content Generation
AI-powered tools assist teachers in creating and revising lesson plans based on curriculum standards and student performance data. The system generates customized worksheets, projects, and assessments tailored to individual learning levels. Annual lesson plan revisions incorporate insights from previous years and latest educational research.

[Continue with all major features - write 2-3 sentences per feature]


SPECIAL REQUIREMENTS & UNIQUE FEATURES
────────────────────────────────────────

Artificial Intelligence Integration: [Detailed 2-3 sentences about AI capabilities]

Personalization Engine: [2-3 sentences about how personalization works]

Gamification: [2-3 sentences about engagement features]


TECHNICAL CONSIDERATIONS
────────────────────────────────────────

- Cloud-based architecture for scalability and accessibility
- Mobile-responsive design for access from any device
- Integration with existing school management systems
- Compliance with data privacy regulations (GDPR, FERPA)
- [Continue with technical requirements]


EXPECTED OUTCOMES & BENEFITS
────────────────────────────────────────

- Improved student academic performance (measurable through test scores)
- Increased teacher efficiency (reduced planning time by estimated 30%)
- Enhanced parent engagement (tracked through portal usage metrics)
- Data-driven decision making for administrators
- [Continue with 5-7 specific, measurable benefits]


CONSTRAINTS & CONSIDERATIONS
────────────────────────────────────────

Budget: [If mentioned, state range. Otherwise: "To be determined during detailed planning phase"]

Timeline: [If mentioned, state deadline. Otherwise: "To be established based on scope finalization"]

Scale: [Number of users, schools, data volume if mentioned]

Compliance: [Any regulatory requirements mentioned]


FUTURE VISION
────────────────────────────────────────

[Write 2-3 paragraphs about long-term goals, scalability plans, subscription model, global expansion, etc.]


CRITICAL FORMATTING RULES:
✓ Use section headers in CAPS followed by line separator (────)
✓ Write in paragraphs for descriptions (NOT bullet points unless listing)
✓ Each feature gets 2-3 flowing sentences
✓ Professional business document tone
✓ Clear spacing between sections (double line break)
✓ 500-700 words total
✓ NO asterisks, NO markdown - just clean text with line separators

OUTPUT: Professional document following EXACT format above."""

    response = model.generate_content(prompt)
    return response.text.strip()

def generate_clarification_questions(preview, conversation_history, raw_input):
    """
    Generate ONE intelligent question at a time
    Maximum 3-4 total questions across entire conversation
    """
    asked_and_answered = ""
    questions_asked_count = len([item for item in conversation_history if 'question' in item])
    
    for item in conversation_history:
        if 'question' in item and 'answer' in item:
            asked_and_answered += f"Q: {item['question']}\nA: {item['answer']}\n\n"
    
    # Stop after 4 questions max
    if questions_asked_count >= 4:
        return "NO_MORE_QUESTIONS"
    
    prompt = f"""You are a business analyst. Ask ONE critical question to clarify requirements.

ORIGINAL CLIENT INPUT:
{raw_input}

DETAILED PREVIEW:
{preview}

ALREADY CLARIFIED ({questions_asked_count} questions asked so far):
{asked_and_answered if asked_and_answered else "Nothing yet"}

TASK: What is the SINGLE most important missing piece of information?

CRITICAL RULES:
1. Ask ONLY ONE question
2. Be specific and direct
3. DON'T repeat what's already known or asked
4. After {4 - questions_asked_count} more questions, you must stop
5. If EVERYTHING essential is clear, return exactly: "NO_MORE_QUESTIONS"

ESSENTIAL INFO CHECKLIST:
□ Budget (specific or rough range)
□ Timeline (deadline or duration)
□ Scale (number of users/students/transactions)
□ Integration needs (existing systems to connect)

If you already know all 4 items above from input or conversation, return: NO_MORE_QUESTIONS

OUTPUT: Single direct question OR "NO_MORE_QUESTIONS"

Example good questions:
"What is your budget range for this project?"
"When do you need this system operational?"
"How many students will use the system initially?"
"What existing systems need to integrate with this ERP?"

Be concise and specific."""

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

def generate_concept_note(preview, clarifications, internal_matches, external_solutions, highlight_points):
    """
    Generate IMPRESSIVE, CLIENT-READY concept note
    Based on detailed preview + clarifications + recommendations
    """
    prompt = f"""You are writing a professional, impressive concept note for a client presentation.

DETAILED PROJECT PREVIEW (your foundation):
{preview}

CLARIFICATIONS RECEIVED:
{clarifications}

KEY POINTS TO EMPHASIZE:
{highlight_points if highlight_points else "Standard emphasis"}

RECOMMENDED INTERNAL SOLUTIONS:
{internal_matches}

RECOMMENDED EXTERNAL TECHNOLOGIES:
{external_solutions}

TASK: Transform all this information into an IMPRESSIVE, COMPREHENSIVE concept note.

USE THIS EXACT STRUCTURE:

═══════════════════════════════════════════════════════════
                        CONCEPT NOTE
            [Generate compelling project title from preview]
═══════════════════════════════════════════════════════════

1. PROJECT OVERVIEW
[2-3 well-written paragraphs explaining what, why, and for whom. Be compelling and clear. Draw from preview but make it flow beautifully.]

2. OBJECTIVES
[List 5-8 specific, measurable objectives. Start with action verbs. Make them SMART where possible.]

3. TARGET USERS & STAKEHOLDERS
[Describe each user group (2-3 sentences each) and their specific needs/benefits]

4. CORE MODULES & FEATURES
[Organize into logical sections. For each major module:]

A. [Module Name]
- [Feature 1] – [Brief description of capability and benefit]
- [Feature 2] – [Brief description]
- [etc.]

B. [Next Module]
[Continue...]

[Cover ALL features mentioned in preview comprehensively]

5. AI-POWERED & INTELLIGENT FEATURES
[If applicable - highlight automation, personalization, AI capabilities in detail]

6. TECHNICAL APPROACH

Architecture Overview:
[Describe high-level technical architecture]

Recommended Technologies:
Internal Components:
[List recommended internal solutions with brief rationale]

External Technologies & Tools:
[List recommended external tools with brief rationale]

Integration Strategy:
[How components work together]

7. IMPLEMENTATION ROADMAP

Phase 1: [Name] (Duration: X months)
- [Key deliverables]

Phase 2: [Name] (Duration: X months)
- [Key deliverables]

[Create 3-5 realistic phases]

8. EXPECTED OUTCOMES & SUCCESS METRICS
- [Specific measurable outcome 1]
- [Specific measurable outcome 2]
[List 5-7 outcomes]

9. BUDGET ESTIMATE
[If budget info available: provide range with breakdown]
[If not available: state "To be determined based on detailed technical specifications and vendor selection"]

10. RISK MITIGATION
[List 3-4 potential risks and mitigation strategies]

11. NEXT STEPS
1. [Immediate action]
2. [Next action]
3. [etc.]

═══════════════════════════════════════════════════════════

CRITICAL QUALITY REQUIREMENTS:
✓ COMPREHENSIVE - include ALL details from preview and clarifications
✓ WELL-STRUCTURED - clear hierarchy, logical flow
✓ PROFESSIONAL TONE - confident, authoritative, client-focused
✓ GRAMMATICALLY PERFECT - zero errors
✓ DETAILED - 2-3 sentences per point, not just bullets
✓ SPECIFIC - use exact terminology, no vague language
✓ IMPRESSIVE - this should wow the client
✓ ACTIONABLE - clear next steps
✓ 800-1200 words total - this is a FULL document

This is a CLIENT PRESENTATION document. Make it impressive, thorough, and professional."""

    response = model.generate_content(prompt)
    return response.text.strip()

def generate_pdf(concept_note_text, project_title="Concept Note"):
    """
    Generate PDF from concept note text
    Returns BytesIO object
    """
    buffer = BytesIO()
    
    # Create PDF
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                           topMargin=0.75*inch, bottomMargin=0.75*inch,
                           leftMargin=1*inch, rightMargin=1*inch)
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#667eea',
        spaceAfter=20,
        alignment=TA_CENTER
    )