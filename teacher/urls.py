from django.urls import path, re_path
from . import views 

urlpatterns = [
    path('loginpage', views.login_page, name='loginpage'),
    path('registrationpage', views.registration_page, name='loginpage'),
    path('registration', views.registration, name='loginpage'),
    path('home', views.login_to_home, name='home'),
    path('logout', views.logout, name='logout'),
    path('courses', views.courses, name='courses'),
    path('addcourse', views.addcourse, name='addcourses'),
    path('courses/<variable>', views.coursePage, name='coursepage'),
    
] 
