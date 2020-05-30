from django.contrib import admin
from django.urls import path
from summarizer_app import views

admin.site.site_header = "Summarizer Admin"
admin.site.site_title = "Summarizer Admin Portal"
admin.site.index_title = "Welcome to Summarizer Researcher Portal"

urlpatterns = [
    path('', views.index, name = 'home'),
    path('general_domain', views.general_domain, name = 'summarizer_app.general_domain'),
    path('financial_domain', views.financial_domain, name = 'summarizer_app.financial_domain'),
    path('about', views.about, name = 'summarizer_app.about'),
    path('news_domain', views.news_domain, name = 'summarizer_app.news_domain'),
    path('upload_file', views.upload_file, name = 'summarizer_app.upload_file'),
    path('summarize_text', views.summarize_text, name = 'summarizer_app.summarize_text')
    # path('', views.index, name = 'home'),
]