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


def find_internal_matches(preview, all_clarifications, internal_products):
    """
    Match client needs with internal products/modules
    Returns: Narrative paragraphs explaining how internal solutions fit
    OPTIMIZED: Uses cached text and smart filtering
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
    
    # Step 2: Smart product filtering
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
        
        # Include high-relevance products or top 5
        if relevance_score > 0 or len(relevant_products) < 5:
            char_limit = 3000 if relevance_score > 2 else 1500
            products_content += f"\n\n=== {product.name} ===\n"
            if product.description:
                products_content += f"Description: {product.description}\n"
            products_content += f"Details: {pdf_text[:char_limit]}\n"
            relevant_products.append((product.name, relevance_score))
    
    # Sort by relevance
    relevant_products.sort(key=lambda x: x[1], reverse=True)
    top_products = [p[0] for p in relevant_products[:8]]
    
    # Step 3: Generate narrative recommendations
    prompt = f"""You are a solution architect writing a recommendation report. Analyze the client's needs and explain how our internal solutions can address them.

CLIENT REQUIREMENTS:
{preview}

CLARIFICATIONS:
{all_clarifications}

AVAILABLE INTERNAL PRODUCTS/MODULES:
{products_content}

MOST RELEVANT PRODUCTS (prioritize these): {', '.join(top_products)}

TASK: Write 2-4 flowing paragraphs (250-350 words total) explaining how our internal solutions can be applied to this specific client project.

CRITICAL RULES:
1. Write in connected paragraphs, NOT lists or bullet points
2. Naturally integrate product/module names into sentences
3. Be SPECIFIC about how each solution addresses their actual needs
4. Focus on the MOST RELEVANT products identified above
5. Only mention products that GENUINELY fit their requirements
6. If none fit well, say so honestly and suggest custom development
7. Maximum 4-5 products mentioned - quality over quantity
8. Use confident, consultative language
9. Mention specific features/capabilities from the product details

WRITING STYLE:
- Conversational but professional
- Future-focused ("This would enable...", "By leveraging...")
- Specific to their project context with real details
- Natural flow between ideas
- Include concrete benefits and outcomes

STRUCTURE EXAMPLE:
"For [specific client need from requirements], our [Product Name] provides a proven foundation. The platform's [specific feature from PDF] directly addresses [their challenge], enabling them to [concrete outcome]. This can be customized to support [their specific workflow mentioned in clarifications], which would [measurable benefit].

To complement this, [Another Product] handles [another requirement they mentioned]. Its [specific capability] has been successfully deployed in similar [industry] contexts, reducing [pain point] significantly. The integration between these systems would streamline [process] and improve [metric].

For the [technical/functional requirement] emphasized in their clarifications, leveraging our [Technical Module] ensures [specific benefit]. This component includes [feature from PDF] that directly supports [their goal]. Combined with [integration point], this creates a comprehensive solution for [their stated objective]."

Now write your recommendation paragraphs for this client project:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating internal recommendations: {str(e)}"


def search_external_solutions(preview, all_clarifications):
    """
    Find external tools/modules/services (not competing products)
    Returns: Narrative paragraphs explaining external technologies
    """
    prompt = f"""You are a technology consultant writing a recommendation report. Suggest external tools and services that would enhance this client's project.

CLIENT REQUIREMENTS:
{preview}

CLARIFICATIONS:
{all_clarifications}

TASK: Write 2-4 flowing paragraphs (250-350 words total) describing external technologies, tools, and services that would complement this solution.

CRITICAL RULES:
1. Write in connected paragraphs, NOT lists or bullet points
2. Naturally weave tool/service names into sentences
3. Suggest COMPONENTS/TOOLS/SERVICES, not competing full products
4. Focus on: frameworks, APIs, cloud services, libraries, platforms
5. Explain specifically WHY each fits their project needs
6. Briefly mention cost considerations naturally in context
7. Prioritize modern, well-supported, practical options
8. Recommend 5-7 technologies maximum
9. Be specific about implementation approach

WRITING STYLE:
- Natural narrative flow between related technologies
- Practical and actionable with concrete use cases
- Specific to their technical requirements mentioned
- Future-focused language about implementation
- Brief cost notes integrated naturally (not as separate lines)
- Group related technologies together logically

STRUCTURE EXAMPLE:
"For the payment processing requirements mentioned in the clarifications, integrating Stripe API would provide a robust, secure solution with industry-standard compliance. The platform offers extensive documentation and supports [specific payment method they need], making implementation straightforward. While Stripe operates on a freemium model with transaction fees, the cost structure scales naturally with their business growth, avoiding large upfront investments.

The front-end architecture would benefit significantly from React's component-based approach, paired with Tailwind CSS for rapid, responsive design development. Both technologies are open-source with massive community support, ensuring long-term maintainability and access to pre-built components that can accelerate their [specific timeline]. For the real-time notification features they emphasized, Firebase Cloud Messaging provides reliable push capabilities across web and mobile platforms at no cost for their expected usage volume.

On the infrastructure side, deploying on AWS Lambda or Google Cloud Functions would handle their variable load efficiently through serverless architecture. This approach eliminates the need for constant server management while automatically scaling during [their peak usage pattern]. For their data management needs requiring [specific database feature mentioned], PostgreSQL provides enterprise-grade reliability and the advanced querying capabilities necessary for their [specific use case], all while remaining cost-effective as an open-source solution."

Now write your technology recommendations for this client project:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating external recommendations: {str(e)}"

def generate_concept_note(description, highlight_points, document_content, client_vision, extracted_requirements, solution_design, external_features, implementation_plan, reference_context):
    """
    Generate a concise, professional concept note (2-3 pages) with unique, non-repetitive content per section.
    FIXED: Now uses the actual paragraph-based recommendations properly
    """

    concept_prompt = f"""
Generate a concise and professional concept note (STRICTLY 2-3 pages / 1200-1500 words) with strategic business language and only essential information.

PROJECT INPUTS:
Description: {description}
Highlights: {highlight_points}
Detailed Content: {document_content}

CLIENT'S SOLUTION VISION:
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
1. Analyze the PROJECT INPUTS and DETAILED CONTENT to identify the ACTUAL project/client name
2. Look for mentions of: company names, project names, client organizations, product names
3. DO NOT use generic words like "description", "document", or field labels
4. If a clear client name exists (like "Friends School", "Precise Eye Hospital", "ABC Corporation"), USE IT
5. Create a professional, descriptive title that reflects the REAL project essence and value proposition
6. Format: "[Technology/Solution Type] for [Client Name]" (e.g., "AI-Powered Educational Platform for Friends School")

CRITICAL CONTENT RULES - ABSOLUTE REQUIREMENTS:
❌ NO REPETITION ALLOWED - Each section must contain completely UNIQUE information
❌ NO overlapping content between sections - violating this is unacceptable
❌ Each section serves ONE specific purpose - stay within that purpose
✓ Be highly specific and contextual to the CLIENT'S actual needs
✓ Use professional business language with strategic depth
✓ Include quantifiable metrics where possible
✓ Write with authority and technical precision
✓ Keep paragraphs concise (3-5 sentences maximum)

GENERATE CONCEPT NOTE WITH THIS STRUCTURE:

[Create Compelling Project Title - Professional, Specific, Client-Focused]

1. ABOUT US

Gaude AI Solutions is a leading artificial intelligence innovation company transforming businesses across multiple industries since 2018. Founded by visionary technologists in Kerala, India, we've grown from a partnership firm into a global AI powerhouse with presence across India, UAE, Uzbekistan, and Kenya. Our expertise spans intelligent automation, data-driven decision systems, and AI-powered business transformation – empowering organizations to achieve measurable efficiency, scalability, and innovation excellence.


2. EXECUTIVE SUMMARY

[Write 1 concise paragraph (100-120 words):
- Opening: The transformative opportunity (ONE sentence)
- The strategic problem (ONE sentence)
- The proposed solution with key differentiator (ONE sentence)
- Expected business impact (ONE sentence)
- Closing: Partnership value (ONE sentence)

Use strategic language. Keep it brief and high-level ONLY.]


3. PROBLEM STATEMENT

[Write 2 concise paragraphs (180-220 words):

Paragraph 1 - Current State & Pain Points (3-4 sentences):
- Client's current situation with context (industry, scale)
- 2-3 specific challenges they face based on the CLIENT'S SOLUTION VISION and REQUIREMENTS
- Business impact of these challenges

Paragraph 2 - Strategic Urgency (2-3 sentences):
- Cost of inaction
- Why this matters NOW based on their actual requirements

❌ DO NOT mention ANY solutions here - only problems from their requirements]


4. PROPOSED SOLUTION

[Write 2-3 paragraphs (250-300 words):

CRITICAL: Use the INTERNAL SOLUTIONS ANALYSIS provided above. This section should explain WHAT the solution is and HOW it will be implemented using our internal products.

If the INTERNAL SOLUTIONS ANALYSIS contains paragraph-based recommendations (not bullet points), use those paragraphs DIRECTLY here. Integrate them naturally into your solution description.

Paragraph 1 - Solution Overview (3-4 sentences):
- What comprehensive solution we're proposing
- The overall technical approach and platform
- Key technologies and accessibility

Paragraph 2 - Internal Solutions Integration (This is where you use the INTERNAL SOLUTIONS content):
[Take the paragraph-based content from INTERNAL SOLUTIONS ANALYSIS and integrate it here. If it's already in paragraph form, use it directly. If it's in bullet form, convert to flowing paragraphs explaining how our internal products/modules address their needs.]

Paragraph 3 - Strategic Positioning (2-3 sentences):
- Competitive advantage created
- Unique differentiators based on our solutions

Use future-focused language: "will transform," "will serve as," "will leverage"

❌ DO NOT list features here - features go in section 5]


5. KEY FEATURES AND FUNCTIONALITIES

[Write in structured format (350-400 words):

Introduction (1 sentence):
"The solution offers comprehensive features designed to address [client's specific needs from requirements]."

Then list 6-8 ESSENTIAL features only based on REQUIREMENTS ANALYSIS and EXTERNAL TECHNOLOGIES:

For [Primary User Group from requirements]:
• Feature Name: 2 sentences (what it does, why it matters to THEIR project)
• Feature Name: 2 sentences (specific to their needs)

For [Secondary User Group]:
• Feature Name: 2 sentences
• Feature Name: 2 sentences

For Administrators/Management:
• Feature Name: 2 sentences

Advanced Capabilities:
• AI-Powered Feature: 2 sentences (if mentioned in requirements)
• Integration Feature: 2 sentences (from EXTERNAL TECHNOLOGIES if relevant)

Draw specific features from REQUIREMENTS and EXTERNAL TECHNOLOGIES.
❌ Only include CRITICAL features mentioned in their requirements, no generic features]


6. IMPLEMENTATION APPROACH AND DEVELOPMENT PROCESS

[Write ONLY phased rollout - NO methodology paragraphs (150-180 words):

"The implementation will follow a strategic phased approach ensuring seamless adoption:"

Phase I - [Foundation/Core Setup] (X weeks/months based on their timeline):
• Scope: [What core components will be built - specific to their project]
• Key Deliverables: [Specific outputs relevant to their needs]

Phase II - [Enhancement/Integration] (X weeks/months):
• Scope: [What additional features/integrations will be added]
• Key Deliverables: [Specific outputs]

Phase III - [Optimization/Scaling] (X weeks/months):
• Scope: [What final optimizations and scaling work]
• Key Deliverables: [Specific outputs]

Closing: "This phased approach ensures iterative feedback, risk mitigation, and seamless stakeholder adoption."

❌ NO methodology paragraphs - phases ONLY based on IMPLEMENTATION PLAN]


7. EXPECTED OUTCOMES AND BENEFITS

[Write 2 paragraphs (200-240 words):

Paragraph 1 - Quantifiable Outcomes (3-4 sentences):
- Specific ROI and efficiency metrics based on their requirements
- Revenue opportunities or cost savings mentioned
- Scalability benefits for their context
- Time savings or process improvements

Paragraph 2 - Stakeholder Benefits (4-5 sentences):
For EACH key stakeholder group from REQUIREMENTS (2-3 groups max):
"[Specific Stakeholder from their requirements] will experience [specific benefits relevant to their needs]."

Keep concise. Use impact language: "significantly improve," "drive measurable results," "enhance"

❌ DO NOT repeat problem statement or solution details]


8. CONCLUSION

[Write 1 paragraph (80-100 words):
- Reaffirm transformative potential specific to their project (1 sentence)
- Partnership value with Gaude AI (1 sentence)
- Next steps and call to action (1-2 sentences)

Keep brief and powerful. NO generic summarization.]


FORMATTING REQUIREMENTS:
✓ Use section numbers (1., 2., 3., etc.)
✓ Section titles in BOLD CAPS
✓ Professional business prose with strategic language
✓ Bullet points with detailed descriptions (• for main, o for sub-points)
✓ Include client-specific terminology from their requirements
✓ NO generic placeholders - use ACTUAL client context
✓ NO brackets or [placeholder text] in final output
✓ Total length: 1200-1500 words (STRICTLY 2-3 pages)
✓ Keep paragraphs concise (2-4 sentences each)
✓ Only include ESSENTIAL, RELEVANT information
✓ Remove all fluff and unnecessary details

TONE & STYLE GUIDELINES:
✓ Confident and authoritative
✓ Strategic and market-aware
✓ Specific and contextual (never generic)
✓ Future-focused with "will" statements
✓ Professional business language
✓ Technical depth where appropriate based on requirements
✓ Value-driven throughout

ABSOLUTE UNIQUENESS REQUIREMENTS - EACH SECTION HAS ONE PURPOSE:
1. About Us = Company credentials ONLY
2. Executive Summary = High-level strategic overview ONLY (no details)
3. Problem Statement = Problems, challenges, urgency from REQUIREMENTS ONLY (no solutions)
4. Proposed Solution = Technical architecture, platform approach, and HOW our INTERNAL SOLUTIONS will be used (use the paragraph content from INTERNAL SOLUTIONS ANALYSIS)
5. Key Features = Detailed user-facing features from REQUIREMENTS ONLY (no architecture)
6. Implementation = Phased rollout timeline ONLY (no methodology prose)
7. Expected Outcomes = Measurable benefits and ROI ONLY (no problems or features)
8. Conclusion = Forward momentum and next steps ONLY (no summarization)

CRITICAL FOR SECTION 4 (PROPOSED SOLUTION):
- The INTERNAL SOLUTIONS ANALYSIS provided above contains paragraph-based recommendations
- Use those paragraphs DIRECTLY in section 4 to explain how our products will be integrated
- Do NOT convert them to bullet points
- Do NOT summarize them generically
- Keep the detailed, specific content about how each internal solution addresses their needs
- This is the ONLY place where internal solutions should be discussed in detail

❌ CRITICAL: If you find yourself repeating information from a previous section, STOP and write something entirely different. Each section must be completely distinct.

Generate the complete, concise, professional concept note now:
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