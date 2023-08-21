from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Student, Department, Course, Takes
from .service import get_depatment_courses, get_student_courses, get_daily_attendance
from utils.graph import Graph
import os
from utils.facedetection import get_face_encodings
from django.core.files.storage import default_storage, FileSystemStorage
import pickle as pkl
from attendancesystems.settings import BASE_DIR, MEDIA_ROOT
from django.core.files import File
# Create your views here.

graph = Graph()

def index(request):
    return render(request, 'home1.html')

def login_page(request):
    try:
        if 'usernameS' in request.session:
            return redirect(login_to_home)
        else:
            return render(request, 'loginpage.html', {'role':'student'})
    except:
        return render(request, 'error.html')


def logout(request):
    if 'usernameS' in request.session:
        request.session.flush()
    return render(request, 'loginpage.html', {'role':'student'})

def registration_page(request):
    return render(request, 'registrationpage.html', {'role':'student'})

def courses(request):
    try:
        if 'usernameS' not in request.session:
            return redirect(login_page)
            
        student_id = request.session['usernameS']
        dept_id = request.session['deptid']
        #std_courses = Takes.objects.filter(student_id=student_id)
        #context = {'courses':std_courses}

        dept_courses = get_depatment_courses(dept_id)
        std_courses = get_student_courses(student_id)
        data = []

        for course in std_courses:
            data.append((dept_courses.get(course), "courses/"+course))
        return render(request, 'courses.html', {'data':data})
    except:
        return render(request, 'error.html')

def addcourse(request):
    try:
        dept_courses = Course.objects.filter(dept_id=request.session['deptid'])
        context = {'courses':dept_courses}

        if request.method == 'GET':
            return render(request, 'addcourse2.html', context)

        if request.method == 'POST':
            course_id = request.POST['newcourse']
            student_id = request.session['usernameS']
            std_ins = Student.objects.get(student_id=student_id)
            obj = Takes.objects.filter(student_id = student_id, course_id=course_id)
            if len(obj) == 0:
                course_ins = Course.objects.get(course_id=course_id)
                ins = Takes(course_id=course_ins, student_id=std_ins)
                ins.save()
            return render(request, 'addcourse2.html', context)
    except:
        return render(request, 'error.html')


def registration(request):
    try:
        if request.method == 'POST' and request.FILES['photo']:
            name = request.POST['name']
            rollnumber = request.POST['rollnum']
            department = request.POST['department']
            password = request.POST['password']
            repassword = request.POST['repassword']
            mail = request.POST['mail']
            mobile = request.POST['mobile']
            photo = request.FILES['photo']

            if password == repassword and rollnumber != "" and password != "":
                ins =  Student(student_id=rollnumber,std_name=name,department=department, password=password, mobile=mobile, mail=mail)
                ins.save()
                student = Student.objects.all().filter(student_id=rollnumber)[0]

                if photo:
                    student.photo = photo
                    #student.faceEncoding=File(f)
                student.save()
                
                encoded_face_vector = get_face_encodings(photo, is_attendance=False)
                f = open(MEDIA_ROOT + '/encodings/'+ rollnumber +'.pickle','wb')
                pkl.dump(encoded_face_vector[0], f)
                f.close()
                #print(encoded_face_vector[0])
                return render(request, 'registrationmessage.html')

            else:
                print("Please check details entered")
                return redirect(registration_page)
    except:
        return render(request, 'error.html')

def login_to_home(request):
    try:
        if request.method == 'POST':
            rollnumber = request.POST['username']
            password = request.POST['password']

            user_data = Student.objects.get(student_id=rollnumber)
            deptname = Department.objects.get(dept_id=user_data.department)
            user_data.std_name = user_data.std_name.upper()
            user = {'user':user_data, 'deptname':deptname, 'role':'student'}
            if user_data.password == password:
                request.session['usernameS'] = rollnumber
                request.session['deptid'] = user_data.department
                return render(request, 'home1.html', user)
            else:
                return render(request, 'loginpage.html', {'role':'student'})

        if request.method == 'GET':
            if 'usernameS' in request.session:
                rollnumber = request.session['usernameS']
                user_data = Student.objects.get(student_id=rollnumber)
                deptname = Department.objects.get(dept_id=user_data.department)
                user = {'user':user_data, 'deptname':deptname, 'role':'student'}
                return render(request, 'home1.html', user)
            else:
                return render(request, 'loginpage.html')
    except:
        return render(request, 'error.html')

def coursePage(request, variable):
    try:
        if variable == 'home':
            return redirect(login_to_home)
        elif variable == 'addcourse':
            return redirect(addcourse)
        elif variable == 'courses':
            return redirect(courses)

        elif variable == 'attendance':
            dept_id = request.session['deptid']
            dept_courses = get_depatment_courses(dept_id)
            course_name = dept_courses.get(request.session['course_id'])
            
            result = get_daily_attendance(request.session['usernameS'], request.session['course_id'])
            dates = result['dates']
            status = result['status']
            
            #print(request.session['username'], request.session['course_id'])

            if len(dates) > 0 and (len(dates)==(len(status))):
                is_data_exist = True
                #file_path1 = graph.plot_daily_attendance(dates, status)
                #file_path2 = graph.plot_attendance_distribution(status)
                #print(file_path1,file_path2)
                data = zip(dates, status)
                statistics = { 'total_classes':len(dates), 'total_present': status.count(1), 'percentage':round(100*status.count(1)/len(dates),2)}
            else:
                is_data_exist = False
                #file_path1 = None
                #file_path2 = None
                data = None
                statistics ={}
            return render(request, 'attendance.html', {'name':course_name, 'data':data, 'statistics':statistics, 'is_data_exist':is_data_exist})

        else:
            dept_id = request.session['deptid']
            dept_courses = get_depatment_courses(dept_id)
            if variable in dept_courses:
                request.session['course_id'] = variable
            dept_id = request.session['deptid']
            dept_courses = get_depatment_courses(dept_id)
            course_name = dept_courses.get(request.session['course_id'])
            return render(request, 'coursepage.html', {'name':course_name})

    except:
        return render(request, 'error.html')






