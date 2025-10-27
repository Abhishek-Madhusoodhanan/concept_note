from django.db import models
from django.core.validators import FileExtensionValidator

class InternalProduct(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    pdf_file = models.FileField(
        upload_to='products/',
        validators=[FileExtensionValidator(allowed_extensions=['pdf'])]
    )
    # NEW: Cache extracted text to avoid re-processing
    extracted_text = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        # Auto-extract text when PDF is first uploaded
        if self.pdf_file and not self.extracted_text:
            try:
                from .ai_handler import extract_text_from_pdf
                self.extracted_text = extract_text_from_pdf(self.pdf_file)[:5000]  # Limit to 5000 chars
            except Exception as e:
                print(f"Error extracting text from {self.name}: {e}")
        super().save(*args, **kwargs)


class ConceptProject(models.Model):
    session_id = models.CharField(max_length=100, unique=True, db_index=True)
    raw_input = models.TextField()
    formatted_preview = models.TextField(blank=True)
    conversation_history = models.JSONField(default=list)
    final_concept_note = models.TextField(blank=True)
    client_name = models.CharField(max_length=200, blank=True, null=True)
    
    # NEW: Store recommendations to avoid regeneration
    internal_recommendations = models.TextField(blank=True, null=True)
    external_recommendations = models.TextField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['session_id']),
        ]

    def __str__(self):
        return f"Project {self.session_id} - {self.client_name or 'Unnamed'}"