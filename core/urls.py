from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/generate-preview/', views.generate_preview, name='generate_preview'),
    path('api/get-clarifications/', views.get_clarifications, name='get_clarifications'),
    path('api/save-clarification/', views.save_clarification, name='save_clarification'),
    path('api/get-recommendations/', views.get_recommendations, name='get_recommendations'),
    path('api/generate-final-note/', views.generate_final_note, name='generate_final_note'),
    path('api/get-products/', views.get_products, name='get_products'),
    path('api/upload-file/', views.upload_file, name='upload_file'), 
    path('api/download-pdf/', views.download_pdf, name='download_pdf'),
        path('api/get-ai-suggestion/', views.get_ai_suggestion, name='get_ai_suggestion'),

]
