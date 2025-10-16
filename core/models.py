from django.db import models

class InternalProduct(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)  # Made optional
    pdf_file = models.FileField(upload_to='products/')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class ConceptProject(models.Model):
    session_id = models.CharField(max_length=100, unique=True)
    raw_input = models.TextField()  # Client's needs description
    formatted_preview = models.TextField(blank=True)
    conversation_history = models.JSONField(default=list)
    final_concept_note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Project {self.session_id}"