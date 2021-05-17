from django.contrib import admin
from django.urls import path

from vincenzo import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.authorization),
    path('registration/', views.registration),
]
