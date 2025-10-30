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
def initiate_project(request):
    """
    Step 1: Create session and store initial input without generating preview
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            raw_input = data.get('raw_input', '').strip()
            highlight_points = data.get('highlight_points', '')
            pdf_text = data.get('pdf_text', '')
            
            if not raw_input:
                return JsonResponse({'error': 'Description is required'}, status=400)
            
            # Create session
            session_id = str(uuid.uuid4())[:8]
            
            # Generate intelligent pre-preview questions
            from .ai_handler import generate_pre_preview_questions
            questions = generate_pre_preview_questions(raw_input, pdf_text, highlight_points)
            
            # Create project with initial data
            project = ConceptProject.objects.create(
                session_id=session_id,
                raw_input=raw_input,
                uploaded_pdf_text=pdf_text,
                pre_preview_questions=questions
            )
            
            return JsonResponse({
                'success': True,
                'session_id': session_id,
                'questions': questions,
                'message': 'Please answer the clarification questions to improve the preview quality'
            })
            
        except Exception as e:
            import traceback
            print(f"Error in initiate_project: {traceback.format_exc()}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST method required'}, status=405)


@csrf_exempt
def save_pre_preview_answers(request):
    """
    Step 2: Save pre-preview clarification answers
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = data.get('session_id')
            answers = data.get('answers', [])
            
            if not session_id:
                return JsonResponse({'error': 'session_id is required'}, status=400)
            
            project = ConceptProject.objects.get(session_id=session_id)
            
            # FIX: Ensure answers is a list and not None
            if answers is None:
                answers = []
                
            project.pre_preview_answers = answers
            
            # Extract client name from answers if provided
            for answer in answers:
                if isinstance(answer, dict) and answer.get('category') == 'client_identification' and answer.get('value'):
                    project.client_name = answer['value']
                    break
            
            project.save()
            
            return JsonResponse({
                'success': True,
                'message': 'Answers saved successfully'
            })
            
        except ConceptProject.DoesNotExist:
            return JsonResponse({'error': 'Project not found'}, status=404)
        except Exception as e:
            import traceback
            print(f"Error in save_pre_preview_answers: {traceback.format_exc()}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST method required'}, status=405)


@csrf_exempt
def upload_supporting_document(request):
    """
    Upload additional supporting documents during pre-preview clarification
    """
    if request.method == 'POST':
        try:
            session_id = request.POST.get('session_id')
            uploaded_file = request.FILES.get('file')
            
            if not session_id or not uploaded_file:
                return JsonResponse({'error': 'Missing session_id or file'}, status=400)
            
            # Extract text from PDF
            file_text = extract_text_from_pdf(uploaded_file)
            
            # Append to existing PDF text
            project = ConceptProject.objects.get(session_id=session_id)
            existing_text = project.uploaded_pdf_text or ""
            project.uploaded_pdf_text = f"{existing_text}\n\n--- Additional Document: {uploaded_file.name} ---\n{file_text}"
            project.save()
            
            return JsonResponse({
                'success': True,
                'filename': uploaded_file.name,
                'message': 'Supporting document uploaded successfully'
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST method required'}, status=405)


# Update existing generate_preview to use pre-preview data
@csrf_exempt
def generate_preview(request):
    """
    Generate preview - handles both old and new flow with pre-preview clarifications
    """
    if request.method == 'POST':
        # Check for API key
        if not os.getenv("GOOGLE_API_KEY"):
            return JsonResponse({
                'error': 'Server configuration error: Google API key is not set.'
            }, status=500)
        
        try:
            data = json.loads(request.body)
            session_id = data.get('session_id')
            
            print(f"DEBUG: Received session_id: {session_id}")
            
            if not session_id:
                # OLD FLOW: Direct preview generation without pre-clarifications
                raw_input = data.get('raw_input')
                highlight_points = data.get('highlight_points', '')
                
                if not raw_input:
                    return JsonResponse({'error': 'raw_input is required'}, status=400)
                
                session_id = str(uuid.uuid4())[:8]
                
                try:
                    formatted_preview = ai_generate_preview(raw_input, highlight_points)
                    if formatted_preview.startswith("Error:"):
                        return JsonResponse({'error': formatted_preview}, status=500)
                except ResourceExhausted:
                    return JsonResponse({
                        'error': 'Quota exceeded. Please wait a moment and try again.'
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
            
            else:
                # NEW FLOW: Use pre-preview clarifications to enhance preview
                print(f"DEBUG: NEW FLOW - Looking up session: {session_id}")
                
                try:
                    project = ConceptProject.objects.get(session_id=session_id)
                    print(f"DEBUG: Found project: {project.id}")
                except ConceptProject.DoesNotExist:
                    print(f"DEBUG: Project NOT FOUND for session {session_id}")
                    return JsonResponse({
                        'error': f'Project not found for session {session_id}'
                    }, status=404)
                
                # Get the original raw input
                if not project.raw_input or not project.raw_input.strip():
                    print("WARNING: project.raw_input is None or empty")
                    return JsonResponse({
                        'error': 'No initial input found for this project. Please start over.'
                    }, status=400)
                
                original_input = str(project.raw_input).strip()
                print(f"DEBUG: Original input length: {len(original_input)}")
                
                # CRITICAL FIX: Build highlight_points string with clarifications
                highlight_points_enhanced = ""
                
                # Add pre-preview answers as context (not as main input)
                if project.pre_preview_answers and isinstance(project.pre_preview_answers, list):
                    highlight_points_enhanced = "ADDITIONAL CLARIFICATIONS:\n"
                    for answer in project.pre_preview_answers:
                        if isinstance(answer, dict) and answer.get('value'):
                            question_text = answer.get('question', '')
                            answer_value = answer.get('value', '')
                            highlight_points_enhanced += f"{question_text}: {answer_value}\n"
                    print(f"DEBUG: Added {len(project.pre_preview_answers)} clarifications to highlights")
                
                # Add PDF context to highlights if available
                if project.uploaded_pdf_text:
                    pdf_snippet = str(project.uploaded_pdf_text)[:2000]
                    highlight_points_enhanced += f"\n\nSUPPORTING DOCUMENTS:\n{pdf_snippet}"
                    print(f"DEBUG: Added PDF context")
                
                print(f"DEBUG: Calling ai_generate_preview with original input and enhanced highlights")
                print(f"DEBUG: Original input preview: {original_input[:200]}")
                print(f"DEBUG: Highlights preview: {highlight_points_enhanced[:200]}")
                
                # CRITICAL: Call ai_generate_preview with original input and clarifications as highlights
                try:
                    formatted_preview = ai_generate_preview(
                        raw_input=original_input,  # Original description only
                        highlight_points=highlight_points_enhanced  # Clarifications as context
                    )
                    
                    if formatted_preview and formatted_preview.startswith("Error:"):
                        return JsonResponse({'error': formatted_preview}, status=500)
                    
                    print(f"DEBUG: Generated preview length: {len(formatted_preview)}")
                    print(f"DEBUG: Preview starts with: {formatted_preview[:200]}")
                    
                except ResourceExhausted:
                    return JsonResponse({
                        'error': 'Quota exceeded. Please wait a moment and try again.'
                    }, status=429)
                except Exception as ai_error:
                    print(f"DEBUG: AI generation error: {ai_error}")
                    import traceback
                    print(traceback.format_exc())
                    return JsonResponse({
                        'error': f'AI generation failed: {str(ai_error)}'
                    }, status=500)
                
                # Save the generated preview
                project.formatted_preview = formatted_preview
                project.save()
                print(f"DEBUG: Saved preview to database")
                
                return JsonResponse({
                    'session_id': session_id,
                    'preview': formatted_preview
                })
            
        except json.JSONDecodeError as je:
            print(f"DEBUG: JSON decode error: {je}")
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"DEBUG: Exception in generate_preview:")
            print(error_trace)
            return JsonResponse({
                'error': f'An unexpected server error occurred: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'POST method required'}, status=405)

@csrf_exempt
def generate_preview(request):
    """
    Generate preview - handles both old and new flow with pre-preview clarifications
    """
    if request.method == 'POST':
        # Check for API key
        if not os.getenv("GOOGLE_API_KEY"):
            return JsonResponse({
                'error': 'Server configuration error: Google API key is not set.'
            }, status=500)
        
        try:
            data = json.loads(request.body)
            session_id = data.get('session_id')
            
            print(f"DEBUG: Received session_id: {session_id}")
            print(f"DEBUG: Request data keys: {data.keys()}")
            
            if not session_id:
                # OLD FLOW: Direct preview generation without pre-clarifications
                raw_input = data.get('raw_input')
                highlight_points = data.get('highlight_points', '')
                
                print(f"DEBUG: OLD FLOW - raw_input length: {len(raw_input) if raw_input else 0}")
                
                if not raw_input:
                    return JsonResponse({
                        'error': 'raw_input is required'
                    }, status=400)
                
                session_id = str(uuid.uuid4())[:8]
                
                try:
                    formatted_preview = ai_generate_preview(raw_input, highlight_points)
                    if formatted_preview.startswith("Error:"):
                        return JsonResponse({'error': formatted_preview}, status=500)
                except ResourceExhausted:
                    return JsonResponse({
                        'error': 'Quota exceeded. Please wait a moment and try again.'
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
            
            else:
                # NEW FLOW: Use pre-preview clarifications to enhance preview
                print(f"DEBUG: NEW FLOW - Looking up session: {session_id}")
                
                try:
                    project = ConceptProject.objects.get(session_id=session_id)
                    print(f"DEBUG: Found project: {project.id}")
                    print(f"DEBUG: Project raw_input: {project.raw_input[:100] if project.raw_input else 'None'}")
                    print(f"DEBUG: Project pre_preview_answers: {project.pre_preview_answers}")
                except ConceptProject.DoesNotExist:
                    print(f"DEBUG: Project NOT FOUND for session {session_id}")
                    return JsonResponse({
                        'error': f'Project not found for session {session_id}'
                    }, status=404)
                
                # Safely get raw_input with fallback
                enhanced_input = ""
                if project.raw_input:
                    enhanced_input = str(project.raw_input)
                else:
                    print("WARNING: project.raw_input is None or empty")
                    return JsonResponse({
                        'error': 'No initial input found for this project. Please start over.'
                    }, status=400)
                
                print(f"DEBUG: Base enhanced_input length: {len(enhanced_input)}")
                
                # FIX: Safely handle pre_preview_answers - it might be None
                if project.pre_preview_answers and isinstance(project.pre_preview_answers, list):
                    enhanced_input += "\n\nCLARIFICATIONS PROVIDED:\n"
                    for answer in project.pre_preview_answers:
                        # FIX: Add type checking to avoid 'NoneType' subscript error
                        if isinstance(answer, dict) and answer.get('value'):
                            question_text = answer.get('question', '')
                            answer_value = answer.get('value', '')
                            enhanced_input += f"- {question_text}: {answer_value}\n"
                    print(f"DEBUG: Added {len(project.pre_preview_answers)} clarifications")
                else:
                    print(f"DEBUG: No pre_preview_answers or not a list: {project.pre_preview_answers}")
                
                # Add PDF context if available
                if project.uploaded_pdf_text:
                    pdf_snippet = str(project.uploaded_pdf_text)[:2000]
                    enhanced_input += f"\n\nSUPPORTING DOCUMENTS:\n{pdf_snippet}"
                    print(f"DEBUG: Added PDF context (first 2000 chars)")
                
                print(f"DEBUG: Final enhanced_input length: {len(enhanced_input)}")
                
                # Generate enhanced preview
                try:
                    formatted_preview = ai_generate_preview(enhanced_input, "")
                    if formatted_preview and formatted_preview.startswith("Error:"):
                        return JsonResponse({'error': formatted_preview}, status=500)
                except ResourceExhausted:
                    return JsonResponse({
                        'error': 'Quota exceeded. Please wait a moment and try again.'
                    }, status=429)
                except Exception as ai_error:
                    print(f"DEBUG: AI generation error: {ai_error}")
                    import traceback
                    print(traceback.format_exc())
                    return JsonResponse({
                        'error': f'AI generation failed: {str(ai_error)}'
                    }, status=500)
                
                # Save the preview
                project.formatted_preview = formatted_preview
                project.save()
                print(f"DEBUG: Saved preview, length: {len(formatted_preview) if formatted_preview else 0}")
                
                return JsonResponse({
                    'session_id': session_id,
                    'preview': formatted_preview
                })
            
        except json.JSONDecodeError as je:
            print(f"DEBUG: JSON decode error: {je}")
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"DEBUG: Exception in generate_preview:")
            print(error_trace)
            return JsonResponse({
                'error': f'An unexpected server error occurred: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'POST method required'}, status=405)

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
            
            # Check if recommendations already exist (caching)
            if project.internal_recommendations and project.external_recommendations:
                return JsonResponse({
                    'internal': project.internal_recommendations,
                    'external': project.external_recommendations,
                    'cached': True
                })
            
            # Generate new recommendations
            internal_products = InternalProduct.objects.all()
            
            all_clarifications = "\n".join([
                f"Q: {item['question']}\nA: {item['answer']}"
                for item in project.conversation_history
            ])
            
            # Add error handling for each recommendation type
            try:
                internal = find_internal_matches(
                    project.formatted_preview,
                    all_clarifications,
                    internal_products
                )
            except Exception as e:
                print(f"Internal recommendations error: {e}")
                internal = "Unable to generate internal recommendations at this time. Please try again."
            
            try:
                external = search_external_solutions(
                    project.formatted_preview,
                    all_clarifications
                )
            except Exception as e:
                print(f"External recommendations error: {e}")
                external = "Unable to generate external recommendations at this time. Please try again."
            
            # Cache the recommendations
            project.internal_recommendations = internal
            project.external_recommendations = external
            project.save()
            
            return JsonResponse({
                'internal': str(internal) if internal else "No internal recommendations available.",
                'external': str(external) if external else "No external recommendations available.",
                'cached': False
            })
            
        except ConceptProject.DoesNotExist:
            return JsonResponse({
                'error': f'Project not found for session {session_id}'
            }, status=404)
        except Exception as e:
            import traceback
            print(f"Error in get_recommendations: {traceback.format_exc()}")
            return JsonResponse({
                'internal': 'Error generating internal recommendations',
                'external': 'Error generating external recommendations',
                'error': str(e)
            }, status=500)

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
@csrf_exempt
def get_ai_suggestion(request):
    """
    API endpoint to get AI suggestions for selected text
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = data.get('session_id')
            selected_text = data.get('selected_text', '').strip()
            suggestion_type = data.get('suggestion_type', 'improve')
            multiple = data.get('multiple', False)  # Get multiple options
            
            if not session_id or not selected_text:
                return JsonResponse({
                    'error': 'Missing session_id or selected_text'
                }, status=400)
            
            # Get the project to access full context
            try:
                project = ConceptProject.objects.get(session_id=session_id)
                full_context = project.formatted_preview or project.final_concept_note or ""
            except ConceptProject.DoesNotExist:
                return JsonResponse({
                    'error': 'Project not found'
                }, status=404)
            
            # Validate text length
            if len(selected_text) < 10:
                return JsonResponse({
                    'error': 'Selected text too short. Please select at least 10 characters.'
                }, status=400)
            
            if len(selected_text) > 1000:
                return JsonResponse({
                    'error': 'Selected text too long. Please select less than 1000 characters.'
                }, status=400)
            
            # Import the function
            from .ai_handler import generate_ai_suggestion, generate_multiple_suggestions
            
            # Generate suggestion(s)
            if multiple:
                suggestions = generate_multiple_suggestions(
                    selected_text, 
                    full_context,
                    count=3
                )
                return JsonResponse({
                    'success': True,
                    'suggestions': suggestions,
                    'original': selected_text
                })
            else:
                result = generate_ai_suggestion(
                    selected_text,
                    full_context,
                    suggestion_type
                )
                return JsonResponse(result)
            
        except json.JSONDecodeError:
            return JsonResponse({
                'error': 'Invalid JSON in request body'
            }, status=400)
        except Exception as e:
            import traceback
            print(f"Error in get_ai_suggestion: {traceback.format_exc()}")
            return JsonResponse({
                'error': f'Unexpected error: {str(e)}'
            }, status=500)
    
    return JsonResponse({
        'error': 'POST method required'
    }, status=405)
@csrf_exempt
def get_ai_suggestion(request):
    if request.method == 'POST':
        try:
            from .ai_handler import model
            data = json.loads(request.body)
            selected_text = data.get('selected_text', '').strip()
            suggestion_type = data.get('suggestion_type', 'improve')
            multiple = data.get('multiple', False)

            if not selected_text:
                return JsonResponse({'success': False, 'error': 'No text provided'}, status=400)

            # Build an intelligent prompt for Gemini
            if multiple:
                prompt = f"""
Generate 3 improved variations of the following sentence or paragraph:
"{selected_text}"

Each version should maintain the same meaning but vary in tone or phrasing.
Return only the 3 rewritten options, numbered clearly.
"""
            else:
                prompt = f"""
Rewrite or {suggestion_type} the following text:
"{selected_text}"

Make it concise, natural, and contextually improved, keeping the same intent.
Return only the rewritten version, no explanations.
"""

            response = model.generate_content(prompt)
            text = response.text.strip()

            # Format for multiple suggestions
            if multiple:
                suggestions = []
                for i, line in enumerate(text.split('\n')):
                    if line.strip():
                        suggestions.append({
                            'id': i + 1,
                            'type': suggestion_type,
                            'text': line.strip().lstrip('123.- ')
                        })
                return JsonResponse({'success': True, 'suggestions': suggestions})
            else:
                return JsonResponse({'success': True, 'suggestion': text})

        except Exception as e:
            import traceback
            print("Error in get_ai_suggestion:", traceback.format_exc())
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    else:
        return JsonResponse({'success': False, 'error': 'POST method required'}, status=405)
@csrf_exempt
def chat_edit_assistant(request):
    """
    Endpoint for the Mini Chatbot Assistant inside the UI.
    Handles conversational text refinement requests.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "")
            selected_text = data.get("selected_text", "")
            conversation = data.get("conversation", [])

            if not user_message or not selected_text:
                return JsonResponse({"error": "Both 'message' and 'selected_text' are required."}, status=400)

            from .ai_handler import conversational_edit_suggestion
            ai_reply = conversational_edit_suggestion(user_message, selected_text, conversation)

            return JsonResponse({"reply": ai_reply})
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "POST method required"}, status=405)
