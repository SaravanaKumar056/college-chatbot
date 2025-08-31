from django.contrib import admin
from django.urls import path, include
from chatbot import views  # Add this line

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.chatbot_page, name='home'),  # Add this line to map '/' to chatbot
    path('bot/', include('chatbot.urls')),
    path('', include('chatbot.urls')),
    path("chatbot/", include("chatbot.urls")),
]
