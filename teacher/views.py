from django.shortcuts import render, redirect
from django.http import HttpResponse
from Student.models import Instructor, Department, Course, Takes, Teaches, Student, DailyAttendance
from .service import get_depatment_courses, get_teacher_courses, get_department_students, get_students_per_course, get_attendance_register, correct_attendance_list
from django.core.files.storage import default_storage, FileSystemStorage
from utils.facedetection import get_face_encodings, get_minimum_similiarity
from attendancesystems.settings import BASE_DIR, MEDIA_ROOT
import os, cv2
from datetime import datetime
import pickle as pkl
import numpy as np
from utils.assignments import hungarian_algorithm, ans_calculation


# Create your views here.
sim_threshold = 0.3

def login_page(request):
    if 'usernameT' in request.session:
        return redirect(login_to_home)
    else:
        return render(request, 'loginpage.html', {'role':'teacher'})

def registration_page(request):
    return render(request, 'registrationpage.html', {'role':'teacher'})


def registration(request):
    try:
        if request.method == 'POST':
            name = request.POST['name']
            department = request.POST['department']
            password = request.POST['password']
            repassword = request.POST['repassword']
            mail = request.POST['mail']
            mobile = request.POST['mobile']
            if password == repassword and name != "" and password != "":
                ins =  Instructor(mail=mail, instructor_name=name,department=department, password=password, mobile=mobile)
                ins.save()
                print("Data has been writteen to the db")
                return render(request, 'registrationmessage.html')
            else:
                print("Please check details entered")
                return redirect(registration_page)
    except:
        return render(request, 'error.html')


def login_to_home(request):

    try:
        if request.method == 'POST':
            mail = request.POST['username']
            password = request.POST['password']

            user_data = Instructor.objects.get(mail=mail)
            deptname = Department.objects.get(dept_id=user_data.department)
            user = {'user':user_data, 'deptname':deptname, 'role':'teacher'}
            if user_data.password == password:
                request.session['usernameT'] = mail
                request.session['deptidT'] = user_data.department
                return render(request, 'home1.html', user)
            else:
                return render(request, 'loginpage.html')

        if request.method == 'GET':
            if 'usernameT' in request.session:
                mail = request.session['usernameT']
                user_data = Instructor.objects.get(mail=mail)
                deptname = Department.objects.get(dept_id=user_data.department)
                user = {'user':user_data, 'deptname':deptname, 'role':'teacher'}
                return render(request, 'home1.html', user)
            else:
                return render(request, 'loginpage.html')
    except:
        return render(request, 'error.html')
        

def logout(request):
    if 'usernameT' in request.session:
        request.session.flush()
    return render(request, 'loginpage.html')


def courses(request):
    if 'usernameT' not in request.session:
        return redirect(login_page)
        
    mail = request.session['usernameT']
    dept_id = request.session['deptidT']
    #std_courses = Takes.objects.filter(student_id=student_id)
    #context = {'courses':std_courses}

    dept_courses = get_depatment_courses(dept_id)
    teacher_courses = get_teacher_courses(mail)
    data = []

    for course in teacher_courses:
        data.append((dept_courses.get(course), "courses/"+course))
    return render(request, 'courses.html', {'data':data})


def addcourse(request):
    
    try:
        if request.method == 'GET':
            deptid = request.session['deptidT']
            dept_courses = Course.objects.filter(dept_id = deptid)
            return render(request, 'teacher_addcourses.html', {'courses':dept_courses})

        if request.method == 'POST':
            course_name = request.POST['coursename']
            mail = request.session['usernameT']
            deptid = request.session['deptidT']
            dept_courses = get_depatment_courses(deptid)

            if course_name in dept_courses.keys():
                teacher_ins = Instructor.objects.filter(mail=mail)
                course_ins = Course.objects.filter(course_id=course_name)
                ins= Teaches(mail=teacher_ins[0], course_id=course_ins[0])
                ins.save()


            else:
                course_id = request.POST['courseid']
                dept_courses = Course.objects.filter(dept_id = deptid)
                dept_ins = Department.objects.filter(dept_id = deptid)
            
            
                obj = Course.objects.filter(course_id=course_id, dept_id=dept_ins[0])
                # print(len(obj))

                if len(obj) == 0:
                    ins1 = Course(course_id=course_id, course_name=course_name, dept_id=dept_ins[0])
                    ins1.save()
                    teacher_ins = Instructor.objects.filter(mail=mail)
                    course_ins = Course.objects.filter(course_id=course_id)
                    ins2 = Teaches(mail=teacher_ins[0], course_id=course_ins[0])
                    ins2.save()
            return render(request, 'teacher_addcourses.html', {'courses':dept_courses})

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
            if request.method == 'GET':
                dept_id = request.session['deptidT']
                course_id = request.session['course_idT']
                dept_courses = get_depatment_courses(dept_id)
                course_name = dept_courses.get(course_id)
                name_attendance = get_attendance_register(course_id, dept_id)
                #print(name_attendance)

                
                dates = []
                for value in list(name_attendance.values())[0]:
                    dates.append(value[0])

                for name, att_list in name_attendance.items():
                    if len(att_list) != len(dates):
                        name_attendance[name] = correct_attendance_list(name_attendance[name], dates)

                #print(dates)
                return render(request, 'attendanceregister.html', {'name':course_name, 'data':name_attendance, 'dates':dates} )

            # if request.method == 'POST':
            #     date = request.POST['date']
            #     dept_id = request.session['deptidT']
            #     course_id = request.session['course_idT']
            #     dept_courses = get_depatment_courses(dept_id)
            #     course_name = dept_courses.get(course_id)
            #     result = get_attendance_register(course_id, date, dept_id)
            #     return render(request, 'attendanceregister.html', {'name':course_name, 'date':date, 'data':result} )
            

        elif variable == 'addstudents':
            return HttpResponse('<h2>Add Students</h2>')
            


        elif variable == 'students':
            dept_id = request.session['deptidT']
            course_id = request.session['course_idT']
            students = get_students_per_course(course_id, dept_id)
            #print(students)
            dept_courses = get_depatment_courses(dept_id)
            course_name = dept_courses.get(course_id)
            return render(request, 'students.html',{'name':course_name, 'data':students} )

        elif variable == 'takeattendance':
            if request.method == 'GET':
                dept_id = request.session['deptidT']
                course_id = request.session['course_idT']
                dept_courses = get_depatment_courses(dept_id)
                course_name = dept_courses.get(course_id)
                return render(request, 'takeattendance.html',{'name':course_name} )

            if request.method == 'POST' and 'photo' in request.FILES:
                #folder='classroom_images/' 
                dept_id = request.session['deptidT']
                course_id = request.session['course_idT']
                dept_courses = get_depatment_courses(dept_id)
                course_name = dept_courses.get(course_id)

                file = request.FILES['photo']


                ext = file.name.split('.')[-1]
                temp = str(datetime.today()).split(':')
                folder_name = "%s_%s_%s_%s" % (dept_id, course_id, temp[0].replace(" ", "_"),temp[1])
                image_save_path = os.path.join(MEDIA_ROOT, "attendance", folder_name)

                try:
                    os.mkdir(image_save_path)
                except:
                    pass

                file_name = default_storage.save(os.path.join(image_save_path, 'classroom_image.'+ext), file)
                file = default_storage.open(file_name)
                # file_url = default_storage.url(file_name)
                # fs = FileSystemStorage(location=folder) 
                # filename = fs.save(file.name, file)
                # file = fs.open(filename)
                # file_url = fs.url(filename)
                classroom_face_vectors = get_face_encodings(file, save_path=image_save_path)
                #print(encoded_face_vectors)
                course_ins = Course.objects.filter(course_id=course_id)
                date_obj = datetime.now().date()
                students = get_students_per_course(course_id, dept_id)

                student_ids = []
                for student in students:
                    student_ids.append(student[1])
                
                num_students_in_course = len(student_ids)
                number_faces_detected = len(classroom_face_vectors)
                cout_present = 0
                count_absent = 0
                
                index_id = {}
                scores_matrix = []

                for i, id in enumerate(student_ids):
                    f = open(MEDIA_ROOT + '/encodings/'+ id +'.pickle', 'rb')
                    student_face_vector = pkl.load(f)
                    f.close()
                    index_id[i] = id
                    # print(f'student_id:{id}')

                    scores = get_minimum_similiarity(student_face_vector, classroom_face_vectors)
                    scores_matrix.append(scores)

                # print(index_id)
                #print(scores_matrix)
                scores_arr = np.array(scores_matrix)
                #print(scores_arr)
                assignments = hungarian_algorithm(scores_arr.copy())
                # print(assignments)
                ans, ans_mat = ans_calculation(scores_arr, assignments)
                print(ans_mat)

                for (row, col) in assignments:
                    score = ans_mat[row][col]
                    student_id = index_id.get(row)
                    del index_id[row]

                    if score < sim_threshold:
                        student_ins = Student.objects.filter(student_id=student_id)
                        ins = DailyAttendance(student_id=student_ins[0], course_id=course_ins[0], date=date_obj, status=True)
                        ins.save()
                        cout_present += 1
                    else:
                        student_ins = Student.objects.filter(student_id=student_id)
                        ins = DailyAttendance(student_id=student_ins[0], course_id=course_ins[0], date=date_obj, status=False)
                        ins.save()
                        count_absent += 1

                # account for missing faces
                for student_id in index_id.values():
                    student_ins = Student.objects.filter(student_id=student_id)
                    ins = DailyAttendance(student_id=student_ins[0], course_id=course_ins[0], date=date_obj, status=False)
                    ins.save()
                    count_absent += 1

                data = [number_faces_detected, num_students_in_course, cout_present, count_absent]
                return render(request, 'take_attendance_message.html',{'name':course_name, 'data':data})

        elif variable == 'webcam_attendance':
            dept_id = request.session['deptidT']
            course_id = request.session['course_idT']
            dept_courses = get_depatment_courses(dept_id)
            course_name = dept_courses.get(course_id)


            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frame = cap.read()
                cv2.imshow('Take Attendance',frame)
    
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

            # file = frame.copy()

            temp = str(datetime.today()).split(':')
            folder_name = "%s_%s_%s_%s" % (dept_id, course_id, temp[0].replace(" ", "_"),temp[1])
            image_save_path = os.path.join(MEDIA_ROOT, "attendance", folder_name)

            try:
                os.mkdir(image_save_path)
            except:
                pass
           
            cv2.imwrite(image_save_path+'/classroom_image.jpg', frame)
            file = image_save_path+'/classroom_image.jpg'
            classroom_face_vectors = get_face_encodings(file, save_path=image_save_path)
            # print(len(classroom_face_vectors))
            course_ins = Course.objects.filter(course_id=course_id)
            date_obj = datetime.now().date()
            students = get_students_per_course(course_id, dept_id)

            student_ids = []
            for student in students:
                student_ids.append(student[1])
                
            num_students_in_course = len(student_ids)
            number_faces_detected = len(classroom_face_vectors)
            cout_present = 0
            count_absent = 0
                
            index_id = {}
            scores_matrix = []

            for i, id in enumerate(student_ids):
                f = open(MEDIA_ROOT + '/encodings/'+ id +'.pickle', 'rb')
                student_face_vector = pkl.load(f)
                f.close()
                index_id[i] = id
                #print(f'student_id:{id}')

                scores = get_minimum_similiarity(student_face_vector, classroom_face_vectors)
                scores_matrix.append(scores)

            # print(index_id)
            #print(scores_matrix)
            scores_arr = np.array(scores_matrix)
            # print(scores_arr)
            assignments = hungarian_algorithm(scores_arr.copy())
            #print(assignments)
            ans, ans_mat = ans_calculation(scores_arr, assignments)
            print(ans_mat)

            for (row, col) in assignments:
                score = ans_mat[row][col]
                student_id = index_id.get(row)
                del index_id[row]

                if score < sim_threshold:
                    student_ins = Student.objects.filter(student_id=student_id)
                    ins = DailyAttendance(student_id=student_ins[0], course_id=course_ins[0], date=date_obj, status=True)
                    ins.save()
                    cout_present += 1
                else:
                    student_ins = Student.objects.filter(student_id=student_id)
                    ins = DailyAttendance(student_id=student_ins[0], course_id=course_ins[0], date=date_obj, status=False)
                    ins.save()
                    count_absent += 1
            
            # account for missing faces
            for student_id in index_id.values():
                student_ins = Student.objects.filter(student_id=student_id)
                ins = DailyAttendance(student_id=student_ins[0], course_id=course_ins[0], date=date_obj, status=False)
                ins.save()
                count_absent += 1


            data = [number_faces_detected, num_students_in_course, cout_present, count_absent]
            return render(request, 'take_attendance_message.html',{'name':course_name, 'data':data})

            # return render(request, 'error.html')
            

        else:
            dept_id = request.session['deptidT']
            dept_courses = get_depatment_courses(dept_id)
            if variable in dept_courses:
                request.session['course_idT'] = variable

            #dept_id = request.session['deptidT']
            #dept_courses = get_depatment_courses(dept_id)
            course_name = dept_courses.get(request.session['course_idT'])
            return render(request, 'teacher_coursepage.html', {'name':course_name})
    except:
        return render(request, 'error.html')
