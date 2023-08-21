from django.db import models
from PIL import Image
import os
import pickle as pkl
# Create your models here.


def get_file_path(instance, filename):
        ext = filename.split('.')[-1]
        filename = "%s.%s" % (instance.student_id, ext)
        return os.path.join('images/', filename)

class Student(models.Model):
    class Meta:
        db_table = 'student'
    student_id = models.CharField(max_length=50, primary_key = True)
    std_name = models.CharField(max_length=50)
    photo = models.ImageField(upload_to=get_file_path, default='', null=True)
    department = models.CharField(max_length=50)
    password = models.CharField(max_length=50)
    mobile = models.CharField(max_length=15)
    mail = models.CharField(max_length=50)
    #faceEncoding = models.FileField(upload_to='encodings/', default='', null=True)


    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        if self.photo:
            img = Image.open(self.photo.path)
            img.save(self.photo.path)


class DailyAttendance(models.Model):
    class Meta:
        db_table = 'dailyattendance'
    student_id = models.ForeignKey("Student", on_delete=models.CASCADE)
    course_id = models.ForeignKey("Course", on_delete=models.CASCADE)
    date = models.DateField(auto_now=False, auto_now_add=False)
    status = models.BooleanField()

class Course(models.Model):
    class Meta:
        db_table = 'course'
    course_id = models.CharField(primary_key=True, max_length=50)
    course_name = models.CharField(max_length=50)
    dept_id = models.ForeignKey("Department", on_delete=models.CASCADE)

class Takes(models.Model):
    class Meta:
        db_table = 'takes'
    student_id = models.ForeignKey("Student", on_delete=models.CASCADE)
    course_id = models.ForeignKey("Course", on_delete=models.CASCADE)
    grade = models.CharField(max_length=5)
    attendace_percentage = models.DecimalField(max_digits=3, decimal_places=0, default=100)

class Instructor(models.Model):
    class Meta:
        db_table = 'instructor'
    mail = models.CharField(primary_key = True, max_length=50)
    instructor_name = models.CharField(max_length=50)
    mobile = models.CharField(max_length=15)
    department = models.CharField(max_length=50)
    password = models.CharField(max_length=50)


class Teaches(models.Model):
    class Meta:
        db_table = 'teaches'
    mail = models.ForeignKey("Instructor", on_delete=models.CASCADE)
    course_id = models.ForeignKey("Course", on_delete=models.CASCADE)
    num_students = models.IntegerField(null=True)

class Department(models.Model):
    class Meta:
        db_table = 'department'
    dept_id = models.CharField(primary_key=True, max_length=5)
    dept_name = models.CharField(max_length=50)
    