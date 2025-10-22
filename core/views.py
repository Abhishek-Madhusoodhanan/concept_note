from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ConceptProject, InternalProduct
import json
import uuid
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import os
from dotenv import load_dotenv
load_dotenv()

from .ai_handler import (
    generate_preview as ai_generate_preview,
    generate_clarification_questions,
    generate_concept_note,
    generate_pdf,
    extract_text_from_pdf,
    process_audio_with_gemini,
    find_internal_matches,
    search_external_solutions
)

def index(request):
    return render(request, 'chat.html')

@csrf_exempt
def upload_audio(request):
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES.get('audio')
            if not uploaded_file:
                return JsonResponse({'error': 'No audio file provided'}, status=400)
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
    if request.method == 'GET':
        try:
            products = InternalProduct.objects.all()
            products_list = [
                {'id': p.id, 'name': p.name, 'description': p.description[:100] if p.description else ''}
                for p in products
            ]
            return JsonResponse({'products': products_list})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({'error': 'No file provided'}, status=400)
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
    if request.method == 'POST':
        if not os.getenv("GOOGLE_API_KEY"):
            return JsonResponse({'error': 'Server configuration error: Google API key is not set.'}, status=500)
        try:
            data = json.loads(request.body)
            raw_input = data.get('raw_input')
            highlight_points = data.get('highlight_points', '')
            session_id = str(uuid.uuid4())[:8]
            try:
                formatted_preview = ai_generate_preview(raw_input, highlight_points)
                if formatted_preview.startswith("Error:"):
                    return JsonResponse({'error': formatted_preview}, status=500)
            except ResourceExhausted:
                return JsonResponse({
                    'error': 'Quota exceeded. The free tier has a limit of requests per minute. Please wait a moment and try again.'
                }, status=429)
            project = ConceptProject.objects.create(
                session_id=session_id,
                raw_input=raw_input,
                formatted_preview=formatted_preview
            )
            return JsonResponse({
                'session_id': session_id,
                'preview': formatted_preview
            })
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            print(f"An unexpected server error occurred: {e}")
            return JsonResponse({'error': 'An unexpected server error occurred.'}, status=500)

@csrf_exempt
def get_clarifications(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        project = ConceptProject.objects.get(session_id=session_id)
        questions = generate_clarification_questions(
            project.formatted_preview,
            project.conversation_history,
            project.raw_input
        )
        return JsonResponse({'questions': questions})

@csrf_exempt
def save_clarification(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        question = data.get('question')
        answer = data.get('answer')
        project = ConceptProject.objects.get(session_id=session_id)
        project.conversation_history.append({'question': question, 'answer': answer})
        project.save()
        return JsonResponse({'status': 'saved'})

@csrf_exempt
def get_recommendations(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        try:
            project = ConceptProject.objects.get(session_id=session_id)
            internal_products = InternalProduct.objects.all()
            all_clarifications = "\n".join([
                f"Q: {item['question']}\nA: {item['answer']}"
                for item in project.conversation_history
            ])
            internal = find_internal_matches(
                project.formatted_preview,
                all_clarifications,
                internal_products
            )
            external = search_external_solutions(
                project.formatted_preview,
                all_clarifications
            )
            return JsonResponse({
                'internal': str(internal) if internal else "No internal recommendations available.",
                'external': str(external) if external else "No external recommendations available."
            })
        except Exception as e:
            return JsonResponse({
                'internal': 'Error generating internal recommendations',
                'external': 'Error generating external recommendations',
                'error': str(e)
            })

@csrf_exempt
def generate_final_note(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = data.get('session_id')
            selected_internal = data.get('selected_internal', [])
            selected_external = data.get('selected_external', [])

            # ✅ Validate session_id
            if not session_id:
                return JsonResponse({'error': 'Missing session_id'}, status=400)

            # ✅ Ensure project exists
            try:
                project = ConceptProject.objects.get(session_id=session_id)
            except ConceptProject.DoesNotExist:
                return JsonResponse({'error': f'No project found for session_id {session_id}'}, status=404)

            # ✅ Build clarifications text safely
            all_clarifications = "\n".join([
                f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
                for item in project.conversation_history or []
            ])

            # 🎯 FIXED: Extract actual client name intelligently
            from .ai_handler import extract_client_name_from_content
            
            actual_client_name = extract_client_name_from_content(
                project.raw_input or "",
                project.formatted_preview or "",
                project.conversation_history or []
            )
            
            # Save the extracted name for later use
            project.client_name = actual_client_name
            project.save()

            # 🧩 Map existing data into the concept note fields
            # Use the ACTUAL project content, not generic labels
            description = project.raw_input or "No description provided."
            highlight_points = " ".join(selected_internal[:3]) if selected_internal else "Standard emphasis"
            document_content = project.formatted_preview or "No detailed preview available."
            client_vision = all_clarifications or "No clarifications provided yet."
            extracted_requirements = all_clarifications or "No explicit requirements extracted."
            solution_design = "\n".join(selected_internal) if selected_internal else "Solution design to be finalized."
            external_features = "\n".join(selected_external) if selected_external else "External technologies not specified."
            implementation_plan = "Implementation roadmap to be developed collaboratively with the client."
            reference_context = f"Session ID: {session_id}\nClient/Project: {actual_client_name}"

            # ✅ Generate the concept note using your AI function
            concept_note = generate_concept_note(
                description=description,
                highlight_points=highlight_points,
                document_content=document_content,
                client_vision=client_vision,
                extracted_requirements=extracted_requirements,
                solution_design=solution_design,
                external_features=external_features,
                implementation_plan=implementation_plan,
                reference_context=reference_context
            )

            # ✅ Save to DB
            project.final_concept_note = concept_note
            project.save()

            return JsonResponse({
                'session_id': session_id,
                'concept_note': concept_note,
                'client_name': actual_client_name  # Return for frontend use
            })

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)

        except Exception as e:
            import traceback
            print("Error generating final note:", traceback.format_exc())
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST method required'}, status=405)
@csrf_exempt
def download_pdf(request):
    import traceback
    import json
    if request.method not in ['POST', 'GET']:
        return JsonResponse({'error': 'POST or GET method required'}, status=405)
    try:
        if request.method == 'POST':
            try:
                data = json.loads(request.body.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
        else:
            data = request.GET.dict()

        session_id = data.get('session_id', '').strip()
        if not session_id:
            return JsonResponse({'error': 'session_id parameter is required'}, status=400)

        try:
            project = ConceptProject.objects.get(session_id=session_id)
        except ConceptProject.DoesNotExist:
            return JsonResponse({'error': f'Project with session_id "{session_id}" not found'}, status=404)

        concept_note_text = data.get('concept_note', '').strip()
        if not concept_note_text:
            concept_note_text = project.final_concept_note
        if not concept_note_text:
            return JsonResponse({'error': 'No concept note available for this project'}, status=404)

        try:
            # 🎯 FIXED: Use intelligently extracted client name
            client_name = getattr(project, 'client_name', None)
            
            if not client_name:
                # Fallback: Try to extract from the concept note itself
                from .ai_handler import extract_client_name_from_content
                client_name = extract_client_name_from_content(
                    project.raw_input or "",
                    project.formatted_preview or "",
                    project.conversation_history or []
                )
            
            # Generate PDF with proper client name
            pdf_buffer = generate_pdf(concept_note_text, client_name=client_name)
            pdf_buffer.seek(0)
        except Exception as e:
            return JsonResponse({'error': f'PDF generation failed: {str(e)}'}, status=500)

        response = FileResponse(pdf_buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="concept_note_{session_id}.pdf"'
        return response

    except Exception as e:
        print(f"Error in download_pdf: {traceback.format_exc()}")
        return JsonResponse({
            'error': 'An unexpected error occurred while generating the PDF',
            'details': str(e)
        }, status=500)
