from utils.sqlalchemy import SqlAlchemySession
from Student.models  import Student, Course, Takes, Department, Instructor


sql_alchemy_session = SqlAlchemySession()


def get_depatment_courses(department_id):
    values = {'dept_id':department_id}
    statment = "SELECT * FROM course WHERE course.dept_id_id= :dept_id;"
    temp = sql_alchemy_session.get_data_with_values(statment, values)
    result = {}
    for row in temp:
        result[row.course_id] = row.course_name
    return result

def get_teacher_courses(mail):
    values = {'mail':mail}
    statment = "SELECT * FROM teaches WHERE teaches.mail_id= :mail;"
    temp = sql_alchemy_session.get_data_with_values(statment, values)
    result = []
    for row in temp:
        result.append(row.course_id_id)
    return result

def get_department_students(dept_id):
    values = {'dept_id':dept_id}
    statment = "SELECT student.student_id, student.std_name, student.photo FROM student WHERE student.department= :dept_id;"
    results = sql_alchemy_session.get_data_with_values(statment, values)
    student_id_name = {}

    for result in results:
        student_id_name[result.student_id] = (result.std_name, result.photo)

    return student_id_name

def get_students_per_course(course_id, dept_id):
    student_id_name = get_department_students(dept_id)
    values = {'course_id':course_id}
    statment = "SELECT * FROM takes WHERE takes.course_id_id= :course_id;"
    results = sql_alchemy_session.get_data_with_values(statment, values)
    students = []

    for result in results:
        students.append((student_id_name.get(result.student_id_id)[0],result.student_id_id, result.grade,result.attendace_percentage, student_id_name.get(result.student_id_id)[1]))
        
    return students

def get_attendance_register(course_id, dept_id):
    student_id_name = get_students_per_course(course_id, dept_id)
    student_name_attendance = {}

    for data in student_id_name:
        values = {'course_id':course_id, 'student_id':data[1]}
        statment = "SELECT * FROM dailyattendance as d WHERE d.course_id_id= :course_id and d.student_id_id= :student_id;"
        results = sql_alchemy_session.get_data_with_values(statment, values)
        statuses = []
        for result in results:
            if result.status == True:
                status = 'Present'
            else:
                status = 'Absent'
            date = str(result.date.day)+'-'+str(result.date.month)+'-'+str(result.date.year)
            statuses.append((date,status))

        student_name_attendance[data[0]] = statuses

    return student_name_attendance

def correct_attendance_list(att_list, dates):
    dates_in_list = []

    for v in att_list:
        dates_in_list.append(v[0])
    
    for i, date in enumerate(dates):
        if date not in dates_in_list:
            att_list.insert(i,(date, 'Absent'))

    return att_list

