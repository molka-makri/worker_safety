from django.db import models
from django.utils import timezone

# Modèles pour le Système de Surveillance de Sécurité Industrielle

class Module(models.Model):
    """Définit les 7 modules du système"""
    
    STATUS_CHOICES = [
        ('active', 'Actif'),
        ('inactive', 'Inactif'),
        ('error', 'Erreur'),
        ('maintenance', 'Maintenance'),
    ]
    
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=200)
    description = models.TextField()
    icon = models.CharField(max_length=50)
    color = models.CharField(max_length=7, default='#3498db')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Module'
        verbose_name_plural = 'Modules'
    
    def __str__(self):
        return self.name


class Detection(models.Model):
    """Enregistre les détections de chaque module"""
    
    module = models.ForeignKey(Module, on_delete=models.CASCADE, related_name='detections')
    timestamp = models.DateTimeField(auto_now_add=True)
    confidence = models.FloatField(default=0.0)  # 0-1
    count = models.IntegerField(default=0)
    details = models.JSONField(default=dict)  # Données brutes du modèle
    
    class Meta:
        verbose_name = 'Détection'
        verbose_name_plural = 'Détections'
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.module.name} - {self.timestamp}"


class Alert(models.Model):
    """Enregistre les alertes générées"""
    
    SEVERITY_CHOICES = [
        ('low', 'Faible'),
        ('medium', 'Moyen'),
        ('high', 'Élevé'),
        ('critical', 'Critique'),
    ]
    
    module = models.ForeignKey(Module, on_delete=models.CASCADE, related_name='alerts')
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    title = models.CharField(max_length=200)
    description = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    acknowledged = models.BooleanField(default=False)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    acknowledged_by = models.CharField(max_length=100, null=True, blank=True)
    details = models.JSONField(default=dict)
    
    class Meta:
        verbose_name = 'Alerte'
        verbose_name_plural = 'Alertes'
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"[{self.severity.upper()}] {self.title}"
    
    def acknowledge(self, user_name='System'):
        """Marquer l'alerte comme reconnue"""
        self.acknowledged = True
        self.acknowledged_at = timezone.now()
        self.acknowledged_by = user_name
        self.save()


class Metric(models.Model):
    """Enregistre les métriques de performance de chaque module"""
    
    module = models.ForeignKey(Module, on_delete=models.CASCADE, related_name='metrics')
    timestamp = models.DateTimeField(auto_now_add=True)
    precision = models.FloatField(default=0.0)  # 0-1
    recall = models.FloatField(default=0.0)  # 0-1
    map_score = models.FloatField(default=0.0)  # Mean Average Precision
    fps = models.FloatField(default=0.0)  # Frames per second
    processing_time = models.FloatField(default=0.0)  # ms
    f1_score = models.FloatField(default=0.0)  # 0-1
    custom_metrics = models.JSONField(default=dict)
    
    class Meta:
        verbose_name = 'Métrique'
        verbose_name_plural = 'Métriques'
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.module.name} - {self.timestamp}"


class FrameAnalysis(models.Model):
    """Analyse détaillée d'une frame vidéo"""
    
    timestamp = models.DateTimeField(auto_now_add=True)
    frame_number = models.IntegerField()
    duration_ms = models.FloatField()
    frame_data = models.JSONField(default=dict)  # Résultats de tous les modules
    
    class Meta:
        verbose_name = 'Analyse Frame'
        verbose_name_plural = 'Analyses Frame'
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Frame {self.frame_number} - {self.timestamp}"


class SystemConfiguration(models.Model):
    """Configuration du système"""
    
    RESOLUTION_CHOICES = [
        ('480p', '480p'),
        ('720p', '720p'),
        ('1080p', '1080p'),
        ('2K', '2K'),
        ('4K', '4K'),
    ]
    
    confidence_threshold = models.FloatField(default=0.5)  # 0-1
    max_fps = models.IntegerField(default=30)
    resolution = models.CharField(max_length=10, choices=RESOLUTION_CHOICES, default='1080p')
    audio_alerts = models.BooleanField(default=True)
    email_alerts = models.BooleanField(default=True)
    storage_limit_gb = models.IntegerField(default=100)
    retention_days = models.IntegerField(default=30)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Configuration Système'
        verbose_name_plural = 'Configuration Système'
    
    def __str__(self):
        return "Configuration Système"


class ModuleStatus(models.Model):
    """Suivi du statut de chaque module"""
    
    module = models.OneToOneField(Module, on_delete=models.CASCADE, related_name='status_info')
    is_running = models.BooleanField(default=False)
    last_detection = models.DateTimeField(null=True, blank=True)
    detection_count = models.IntegerField(default=0)
    alert_count = models.IntegerField(default=0)
    average_confidence = models.FloatField(default=0.0)
    error_count = models.IntegerField(default=0)
    last_error = models.TextField(null=True, blank=True)
    uptime_percentage = models.FloatField(default=100.0)
    
    class Meta:
        verbose_name = 'Statut Module'
        verbose_name_plural = 'Statuts Module'
    
    def __str__(self):
        return f"Status: {self.module.name}"


class EventLog(models.Model):
    """Journal des événements importants"""
    
    EVENT_TYPES = [
        ('system_start', 'Démarrage Système'),
        ('system_stop', 'Arrêt Système'),
        ('alert_generated', 'Alerte Générée'),
        ('module_error', 'Erreur Module'),
        ('config_changed', 'Configuration Modifiée'),
        ('model_loaded', 'Modèle Chargé'),
    ]
    
    timestamp = models.DateTimeField(auto_now_add=True)
    event_type = models.CharField(max_length=50, choices=EVENT_TYPES)
    module = models.ForeignKey(Module, on_delete=models.SET_NULL, null=True, blank=True)
    description = models.TextField()
    user = models.CharField(max_length=100, default='System')
    details = models.JSONField(default=dict)
    
    class Meta:
        verbose_name = 'Journal Événement'
        verbose_name_plural = 'Journaux Événement'
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.event_type} - {self.timestamp}"
