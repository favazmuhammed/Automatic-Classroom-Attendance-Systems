from django.contrib import admin
from .models import Student, Instructor, Course, Department, Takes, DailyAttendance, Teaches

# Register your models here.
admin.site.register(Student)
admin.site.register(Instructor)
admin.site.register(Course)
admin.site.register(Department)
admin.site.register(Takes)
admin.site.register(DailyAttendance)
admin.site.register(Teaches)