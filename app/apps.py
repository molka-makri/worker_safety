from django.apps import AppConfig


class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'

    def ready(self):
        try:
            from .startup_warmup import start_background_warmup

            start_background_warmup()
        except Exception as exc:
            print(f"[StartupWarmup] WARNING: ready() warmup skipped: {exc}")
