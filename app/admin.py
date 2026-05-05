from django.contrib import admin
from django.utils.html import format_html
from .models import (
    Module, Detection, Alert, Metric, FrameAnalysis,
    SystemConfiguration, ModuleStatus, EventLog
)

# Enregistrement des modèles dans l'admin

@admin.register(Module)
class ModuleAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'status_badge', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at', 'updated_at')
    
    def status_badge(self, obj):
        colors = {
            'active': '#28a745',
            'inactive': '#6c757d',
            'error': '#dc3545',
            'maintenance': '#ffc107',
        }
        return format_html(
            '<span style="color: {}; font-weight: bold;\">{}</span>',
            colors.get(obj.status, '#000'),
            obj.get_status_display()
        )
    status_badge.short_description = 'Statut'


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ('title', 'module', 'severity_badge', 'timestamp', 'acknowledged_badge')
    list_filter = ('severity', 'acknowledged', 'timestamp')
    search_fields = ('title', 'description')
    readonly_fields = ('timestamp', 'acknowledged_at')
    fieldsets = (
        ('Informations', {
            'fields': ('module', 'title', 'description', 'severity')
        }),
        ('Reconnaissance', {
            'fields': ('acknowledged', 'acknowledged_at', 'acknowledged_by')
        }),
        ('Détails', {
            'fields': ('timestamp', 'details'),
            'classes': ('collapse',)
        }),
    )
    
    def severity_badge(self, obj):
        colors = {
            'low': '#17a2b8',
            'medium': '#ffc107',
            'high': '#fd7e14',
            'critical': '#dc3545',
        }
        return format_html(
            '<span style=\"background-color: {}; color: white; padding: 5px 10px; border-radius: 3px;\">{}</span>',
            colors.get(obj.severity, '#000'),
            obj.get_severity_display()
        )
    severity_badge.short_description = 'Sévérité'
    
    def acknowledged_badge(self, obj):
        if obj.acknowledged:
            return format_html(
                '<span style=\"color: #28a745;\">✓ Reconnue</span>'
            )
        return format_html(
            '<span style=\"color: #dc3545;\">✗ Non reconnue</span>'
        )
    acknowledged_badge.short_description = 'Reconnaissance'


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ('module', 'timestamp', 'confidence', 'count')
    list_filter = ('module', 'timestamp')
    search_fields = ('module__name',)
    readonly_fields = ('timestamp', 'details')
    fieldsets = (
        ('Détection', {
            'fields': ('module', 'timestamp', 'confidence', 'count')
        }),
        ('Détails', {
            'fields': ('details',)
        }),
    )


@admin.register(Metric)
class MetricAdmin(admin.ModelAdmin):
    list_display = ('module', 'timestamp', 'precision', 'recall', 'map_score', 'fps')
    list_filter = ('module', 'timestamp')
    search_fields = ('module__name',)
    readonly_fields = ('timestamp',)
    fieldsets = (
        ('Métriques Principales', {
            'fields': ('module', 'precision', 'recall', 'map_score', 'f1_score')
        }),
        ('Performance', {
            'fields': ('fps', 'processing_time')
        }),
        ('Personnalisé', {
            'fields': ('custom_metrics',),
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('timestamp',)
        }),
    )


@admin.register(FrameAnalysis)
class FrameAnalysisAdmin(admin.ModelAdmin):
    list_display = ('frame_number', 'timestamp', 'duration_ms')
    list_filter = ('timestamp',)
    readonly_fields = ('timestamp',)
    fieldsets = (
        ('Frame', {
            'fields': ('frame_number', 'timestamp', 'duration_ms')
        }),
        ('Données', {
            'fields': ('frame_data',)
        }),
    )


@admin.register(SystemConfiguration)
class SystemConfigurationAdmin(admin.ModelAdmin):
    fields = (
        'confidence_threshold', 'max_fps', 'resolution',
        'audio_alerts', 'email_alerts', 'storage_limit_gb',
        'retention_days', 'updated_at'
    )
    readonly_fields = ('updated_at',)


@admin.register(ModuleStatus)
class ModuleStatusAdmin(admin.ModelAdmin):
    list_display = ('module', 'is_running_badge', 'detection_count', 'alert_count', 'uptime_percentage')
    list_filter = ('is_running', 'module')
    readonly_fields = ('last_detection', 'last_error')
    
    def is_running_badge(self, obj):
        color = '#28a745' if obj.is_running else '#6c757d'
        status = 'En cours' if obj.is_running else 'Arrêté'
        return format_html(
            '<span style=\"color: {}; font-weight: bold;\">{}</span>',
            color, status
        )
    is_running_badge.short_description = 'Statut'


@admin.register(EventLog)
class EventLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'event_type', 'module', 'user')
    list_filter = ('event_type', 'timestamp', 'module')
    search_fields = ('description', 'user')
    readonly_fields = ('timestamp',)
    fieldsets = (
        ('Événement', {
            'fields': ('event_type', 'module', 'description', 'user')
        }),
        ('Détails', {
            'fields': ('timestamp', 'details')
        }),
    )
