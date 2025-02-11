import cv2
import numpy as np

# فتح ملف الفيديو
cap = cv2.VideoCapture('020146-3173 (35).mp4')

# التحقق من فتح الفيديو بنجاح
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# قراءة الفيديو إطاراً تلو الآخر
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تحويل الإطار إلى الأبيض والأسود
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # تطبيق هيستوغرام التدرج اللوني
    equalized_frame = cv2.equalizeHist(gray_frame)

    # دمج الصورة الأصلية والصورة بعد التسوية لعرضهما جنبًا إلى جنب
    combined = cv2.hconcat([cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR), cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)])

    # تغيير حجم الصورة المدمجة لتناسب النافذة المطلوبة
    resized_combined = cv2.resize(combined, (1400, 700))

    # عرض الإطارات المدمجة
    cv2.imshow('Original and Equalized Frames', resized_combined)

    # الضغط على 'q' للخروج
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# إغلاق كل شيء
cap.release()
cv2.destroyAllWindows()
