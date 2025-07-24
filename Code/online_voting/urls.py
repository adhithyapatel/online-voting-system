"""online_voting URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from application import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home,name='home'),
    path('admin-login',views.Admin_login_view,name='admin_login'),
    path('register',views.register,name='register'),
    path('login', views.login_view, name='login'),
    path('candidate-login', views.login_view, name='candidatelogin'),
    path('logout',views.logout_view,name='logout'),
    path('voter-registration',views.voter_registration,name='voter_registration'),
    path('candidate-registration',views.candidate_registration,name='candidate_registration'),
    path("add-party/",views.add_party, name="add_party"),
    path("add-election/",views.add_election, name="add_election"),
    path('elections',views.elections,name='elections'),
    path('cast-vote/<int:pk>/',views.cast_vote_made,name='cast_vote'),
    path('vote-made',views.vote_made,name='vote_made'),
    path('results/<int:pk>/',views.result,name='results'),
    path('view-election',views.view_result,name='view_result'),
    path("view-candidates/",views.view_candidates, name="view_candidates"),
]
if settings.DEBUG:
    urlpatterns+=static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)
