import os
import time

path = r"D:\txt\Sistem_Intelijen_Pasar_2_0_Lengkap.docx"
if os.path.exists(path):
    print("File exists!")
    print("Size:", os.path.getsize(path), "bytes")
    print("Last modified:", time.ctime(os.path.getmtime(path)))
else:
    print("File does not exist!")
