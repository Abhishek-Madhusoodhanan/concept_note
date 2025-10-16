from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ConceptProject, InternalProduct
from .ai_handler import *
import json
import uuid
from django.http import FileResponse

def index(request):
    """Main chat interface page"""
    return render(request, 'chat.html')

@csrf_exempt
def upload_audio(request):
    """
    Handle audio file upload and transcribe using Gemini
    """
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES.get('audio')
            
            if not uploaded_file:
                return JsonResponse({'error': 'No audio file provided'}, status=400)
            
            # Process audio with Gemini
            from .ai_handler import process_audio_with_gemini
            
            transcribed_text = process_audio_with_gemini(uploaded_file)
            
            return JsonResponse({
                'success': True,
                'transcribed_text': transcribed_text,
                'filename': uploaded_file.name
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def get_products(request):
    """
    API to fetch all internal products
    """
    if request.method == 'GET':
        try:
            products = InternalProduct.objects.all()
            products_list = []
            
            for product in products:
                products_list.append({
                    'id': product.id,
                    'name': product.name,
                    'description': product.description[:100] if product.description else ''  # First 100 chars
                })
            
            return JsonResponse({'products': products_list})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
@csrf_exempt
def upload_file(request):
    """
    Handle PDF file upload and extract text
    """
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES.get('file')
            
            if not uploaded_file:
                return JsonResponse({'error': 'No file provided'}, status=400)
            
            # Extract text from PDF
            from .ai_handler import extract_text_from_pdf
            
            file_text = extract_text_from_pdf(uploaded_file)
            
            return JsonResponse({
                'success': True,
                'extracted_text': file_text,
                'filename': uploaded_file.name
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)        

@csrf_exempt
def generate_preview(request):
    """
    Step 1: Generate formatted preview from raw input
    WHY: Clean up client's messy input first
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        raw_input = data.get('raw_input')
        highlight_points = data.get('highlight_points', '')
        
        # Generate session ID
        session_id = str(uuid.uuid4())[:8]
        
        # Generate preview using AI
        from .ai_handler import generate_preview as ai_generate_preview
        formatted_preview = ai_generate_preview(raw_input, highlight_points)
        
        # Save to database
        project = ConceptProject.objects.create(
            session_id=session_id,
            raw_input=raw_input,
            formatted_preview=formatted_preview
        )
        
        return JsonResponse({
            'session_id': session_id,
            'preview': formatted_preview
        })

@csrf_exempt
def get_clarifications(request):
    """
    Step 2: AI asks intelligent questions based on context
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        
        project = ConceptProject.objects.get(session_id=session_id)
        
        # Generate questions with FULL context
        from .ai_handler import generate_clarification_questions
        questions = generate_clarification_questions(
            project.formatted_preview,
            project.conversation_history,
            project.raw_input  # ADD THIS - original input
        )
        
        return JsonResponse({'questions': questions})

@csrf_exempt
def save_clarification(request):
    """
    Step 3: Save user's answers
    WHY: Build conversation history for context
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        question = data.get('question')
        answer = data.get('answer')
        
        project = ConceptProject.objects.get(session_id=session_id)
        project.conversation_history.append({
            'question': question,
            'answer': answer
        })
        project.save()
        
        return JsonResponse({'status': 'saved'})

@csrf_exempt
def get_recommendations(request):
    """
    Step 4: Get internal + external recommendations with validation
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        
        try:
            project = ConceptProject.objects.get(session_id=session_id)
            internal_products = InternalProduct.objects.all()
            
            # Format clarifications
            all_clarifications = "\n".join([
                f"Q: {item['question']}\nA: {item['answer']}"
                for item in project.conversation_history
            ])
            
            # Get recommendations
            from .ai_handler import find_internal_matches, search_external_solutions
            
            internal = find_internal_matches(
                project.formatted_preview,
                all_clarifications,
                internal_products
            )
            
            external = search_external_solutions(
                project.formatted_preview,
                all_clarifications
            )
            
            # Ensure strings are returned
            internal_str = str(internal) if internal else "No internal recommendations available."
            external_str = str(external) if external else "No external recommendations available."
            
            return JsonResponse({
                'internal': internal_str,
                'external': external_str
            })
            
        except Exception as e:
            print(f"Error in get_recommendations: {str(e)}")
            return JsonResponse({
                'internal': 'Error generating internal recommendations',
                'external': 'Error generating external recommendations',
                'error': str(e)
            })
        
@csrf_exempt
def generate_final_note(request):
    """
    Step 5: Generate complete concept note
    WHY: Final deliverable document
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        selected_internal = data.get('selected_internal', [])
        selected_external = data.get('selected_external', [])
        
        project = ConceptProject.objects.get(session_id=session_id)
        
        # Format data
        all_clarifications = "\n".join([
            f"Q: {item['question']}\nA: {item['answer']}"
            for item in project.conversation_history
        ])
        
        # Generate concept note
        from .ai_handler import generate_concept_note as ai_generate_note
        concept_note = ai_generate_note(
            project.formatted_preview,
            all_clarifications,
            "\n".join(selected_internal),
            "\n".join(selected_external),
            ""  # highlight points if needed
        )
        
        project.final_concept_note = concept_note
        project.save()
        
        return JsonResponse({
            'concept_note': concept_note,
            'session_id': session_id
        })
    


@csrf_exempt
def download_pdf(request):
    """
    Generate and download concept note as PDF
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = data.get('session_id')
            concept_note = data.get('concept_note')
            
            if not concept_note:
                return JsonResponse({'error': 'No concept note provided'}, status=400)
            
            # Generate PDF
            from .ai_handler import generate_pdf
            
            pdf_buffer = generate_pdf(concept_note, f"Concept Note - {session_id}")
            
            # Return as downloadable file
            response = FileResponse(pdf_buffer, content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="concept_note_{session_id}.pdf"'
            
            return response
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)