from utils.sqlalchemy import SqlAlchemySession
from Student.models  import Student, Course, Takes, Department

sql_alchemy_session = SqlAlchemySession()
months = {
    1:'Jan',
    2:'Feb',
    3:'Mar',
    4:'Apr',
    5:'May',
    6:'Jun',
    7:'Jul',
    8:'Aug',
    9:'Sep',
    10:'Oct',
    11:'Nov',
    12:'Dec',
}

def get_depatment_courses(department_id):
    values = {'dept_id':department_id}
    statment = "SELECT * FROM course WHERE course.dept_id_id= :dept_id;"
    temp = sql_alchemy_session.get_data_with_values(statment, values)
    result = {}
    for row in temp:
        result[row.course_id] = row.course_name
    return result

def get_student_courses(student_id):
    values = {'std_id':student_id}
    statment = "SELECT * FROM takes WHERE takes.student_id_id= :std_id;"
    temp = sql_alchemy_session.get_data_with_values(statment, values)
    result = []
    for row in temp:
        result.append(row.course_id_id)
    return result

def get_daily_attendance(student_id, course_id):
    result ={'dates':[], 'status':[]}
    values = {'std_id':student_id, 'course_id':course_id}
    statment = "SELECT * FROM dailyattendance WHERE dailyattendance.student_id_id= :std_id AND dailyattendance.course_id_id= :course_id;"
    try:
        temp = sql_alchemy_session.get_data_with_values(statment, values)

        for row in temp:
            result['dates'].append(months.get(row.date.month)+'-'+str(row.date.day))
            if row.status == True:
                result['status'].append(1)
            else:
                result['status'].append(0)
    except:
        pass

    return result

