import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import uuid
import io
import boto3
import numpy as np
import os



class Graph:
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        
        
    def plot_daily_attendance(self, dates, status):
        x = np.array([i+1 for i in range(len(dates))])
        y = np.array(status)

        mask1 = y < 0.5
        mask2 = y >= 0.5
        y_1 = [1 for _ in y[mask1]]
        plt.figure(figsize = (15,3))
        plt.bar(x[mask1], y_1, color = 'tomato')
        plt.bar(x[mask2], y[mask2], color = 'forestgreen')
        plt.xticks(x, dates, rotation=45, fontweight='bold')
        plt.tick_params(left = False, right = False , labelleft = False ,bottom = False)
        plt.box(False)
        plt.title('Daily Attendance Status')
        
        file_name = 'attendancesystems\\static\\admin\\img\\dailyattendance.svg'
        file_path = os.path.join(self.BASE_DIR,file_name)
        # file_path = 'http://127.0.0.1:8000/static/graphs/dailyattendance.jpg'
        plt.savefig(file_path, bbox_inches = 'tight')

        return 'admin/img/dailyattendance.svg'

    def plot_attendance_distribution(self, status):
        labels = ['Present', 'Absent']

        count = [status.count(1), status.count(0)]
        colors = ['forestgreen', 'tomato']

        # Creating plot
        fig, ax = plt.subplots(figsize =(10, 7))
        wedges, texts, autotexts = ax.pie(
            count,
            autopct='%1.2f%%',
            colors = colors,
            startangle = 90,
            textprops={'fontsize': 14, 'fontweight':'bold'})

        # Adding legend
        ax.legend(wedges, labels,
                loc ="center left",
                bbox_to_anchor =(1, 0, 0.5, 1),
                fontsize = 'x-large')
        plt.title('Attendance Percentage')
        file_name = 'attendancesystems\\static\\admin\\img\\attendancepercentage.svg'
        file_path = os.path.join(self.BASE_DIR, file_name)
        plt.savefig(file_path, bbox_inches = 'tight')

        return 'admin/img/attendancepercentage.svg'


    