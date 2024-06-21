import cv2
import numpy as np

# فتح ملف الفيديو
cap = cv2.VideoCapture('.mp4')

# التحقق من فتح الفيديو بنجاح
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# قراءة الفيديو إطاراً تلو الآخر
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تطبيق الفلتر الوسيطي
    median_filtered = cv2.medianBlur(frame, 5)

    # دمج الصورة الأصلية والصورة المعدلة لعرضهما جنباً إلى جنب
    combined = cv2.hconcat([frame, median_filtered])

    # تغيير حجم الصورة المدمجة لتتناسب مع النافذة المطلوبة
    resized_combined = cv2.resize(combined, (1400, 700))

    # عرض الإطارات المدمجة
    cv2.imshow('Original and Median Filtered Frame', resized_combined)

    # الضغط على 'q' للخروج
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# إغلاق كل شيء
cap.release()
cv2.destroyAllWindows()
