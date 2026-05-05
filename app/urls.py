from django.urls import path
from .views import (
    IndexView,
    SafetyDashboardView,
    LiveView,
    AlertsView,
    SettingsView,
    ReportsView,
    ModuleDetailView,
    UserLoginView,
    UserLogoutView,
    SignUpView,
    api_test,
    api_module_events,
    api_clear_alerts,
    api_fall_detection,
    api_fall_detection_batch,
    api_fatigue_detection,
    api_fatigue_detection_batch,
    api_ppe_detection,
    api_sign_detect,
    api_spill_detection,
    api_manhole_detection,
    api_manhole_depth,
    api_blocked_exit_detection,
    api_fire_detection,
    api_proximity_detection,
    api_proximity_detection_batch,
    api_posture_detection,   # CAM 9 — posture
    api_panic_detection,     # CAM 10 — panic
    api_worker_tracking_detection,
    api_chat,

)

app_name = 'app'

urlpatterns = [
    # ── Pages ──────────────────────────────────────────────
    path('',                    UserLoginView.as_view(),        name='index'),
    path('dashboard/',          SafetyDashboardView.as_view(),  name='dashboard'),
    path('live/',               LiveView.as_view(),             name='live'),
    path('alerts/',             AlertsView.as_view(),           name='alerts'),
    path('settings/',           SettingsView.as_view(),         name='settings'),
    path('reports/',            ReportsView.as_view(),          name='reports'),
    path('module/<slug:slug>/', ModuleDetailView.as_view(),     name='module_detail'),

    # ── Auth ───────────────────────────────────────────────
    path('login/',              UserLoginView.as_view(),        name='login'),
    path('logout/',             UserLogoutView.as_view(),       name='logout'),
    path('signup/',             SignUpView.as_view(),           name='signup'),

    # ── API — Tests ────────────────────────────────────────
    path('api/test/',           api_test,                       name='api_test'),
    path('api/module-events/<slug:slug>/', api_module_events,   name='api_module_events'),
    path('api/alerts/clear/',   api_clear_alerts,               name='api_clear_alerts'),

    # ── API — Détection de chute ───────────────────────────
    path('api/fall-detection/',        api_fall_detection,        name='api_fall_detection'),
    path('api/fall-detection/batch/',  api_fall_detection_batch,  name='api_fall_detection_batch'),

    # ── API — Détection de fatigue ─────────────────────────
    path('api/fatigue-detection/',       api_fatigue_detection,       name='api_fatigue_detection'),
    path('api/fatigue-detection/batch/', api_fatigue_detection_batch, name='api_fatigue_detection_batch'),

    # ── API — Détection de risques ─────────────────────────
    path('api/ppe-detection/', api_ppe_detection, name='api_ppe_detection'),
    path('api/sign-detect/', api_sign_detect, name='api_sign_detect'),
    path('api/spill-detection/', api_spill_detection, name='api_spill_detection'),
    path('api/manhole-detection/', api_manhole_detection, name='api_manhole_detection'),
    path('api/manhole-depth/', api_manhole_depth, name='api_manhole_depth'),
    path('api/blocked-exit-detection/', api_blocked_exit_detection, name='api_blocked_exit_detection'),
    path('api/fire-detection/', api_fire_detection, name='api_fire_detection'),

    # ── API — Détection de Proximité ───────────────────────
    path('api/proximity-detection/', api_proximity_detection, name='api_proximity_detection'),
    path('api/proximity-detection/batch/', api_proximity_detection_batch, name='api_proximity_detection_batch'),

    # ── API — Posture & Panique (CAM 9 / CAM 10) ──────────
    path('api/posture-detection/', api_posture_detection, name='api_posture_detection'),
    path('api/panic-detection/',   api_panic_detection,   name='api_panic_detection'),
    
    # ── API — Tracking travailleurs (CAM 8) ───────────────
    path('api/worker-tracking-detection/', api_worker_tracking_detection, name='api_worker_tracking_detection'),
     path('api/chat/', api_chat, name='api_chat'),
]
