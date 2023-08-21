from django.urls import path, re_path
from . import views 
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.index, name='index'),
    path('home', views.login_to_home, name='home'),
    path('loginpage', views.login_page, name='loginpage'),
    path('registrationpage', views.registration_page, name='register'),
    path('courses', views.courses, name='courses'),
    path('addcourse', views.addcourse, name='addcourses'),
    path('registration', views.registration, name='registration'),
    path('logout', views.logout, name='logout'),
    path('courses/<variable>', views.coursePage, name='coursepage'),
] 

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)