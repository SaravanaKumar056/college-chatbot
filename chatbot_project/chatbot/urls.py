from django.urls import path
#from .views import chatbot_api 
from django.urls import path
from . import views

urlpatterns = [
    path("chat/", views.chatbot_api, name="chatbot_api"),
    path('', views.chatbot_page, name='chatbot_page'),
    path('chatbot-api/', views.chatbot_api, name='chatbot_api'),
    path("api/", views.chatbot_api, name="chatbot_api"),
]
