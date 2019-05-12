from django.urls import path
from polls import views

app_name = 'polls'

urlpatterns = [
    path('', views.index, name='index'),
    path('polls/<int:question_id>/', views.detail, name='detail'),
    path('polls/<int:question_id>/vote/', views.vote, name='vote'),
    #path('polls/<int:question_id>/results/', views.results, name='results'),
    path('polls/miss/', views.miss, name='miss'),
    path('polls/boho/', views.boho, name='boho'),
    path('polls/search/', views.search, name='search'),
    path('polls/insert/', views.Insert, name='insert'),
    path('polls/results/', views.results, name='results'),
    #path('polls/upload/', views.upload, name='upload')
]
