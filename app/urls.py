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
    api_fall_detection,
    api_fall_detection_batch,
)

app_name = 'app'

urlpatterns = [
    path('', UserLoginView.as_view(), name='index'),
    path('dashboard/', SafetyDashboardView.as_view(), name='dashboard'),
    path('live/', LiveView.as_view(), name='live'),
    path('alerts/', AlertsView.as_view(), name='alerts'),
    path('settings/', SettingsView.as_view(), name='settings'),
    path('reports/', ReportsView.as_view(), name='reports'),
    path('module/<slug:slug>/', ModuleDetailView.as_view(), name='module_detail'),
    path('login/', UserLoginView.as_view(), name='login'),
    path('logout/', UserLogoutView.as_view(), name='logout'),
    path('signup/', SignUpView.as_view(), name='signup'),
    path('api/test/', api_test, name='api_test'),
    path('api/fall-detection/', api_fall_detection, name='api_fall_detection'),
    path('api/fall-detection/batch/', api_fall_detection_batch, name='api_fall_detection_batch'),
]
