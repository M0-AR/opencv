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

    # تطبيق الترشيح الثنائي
    bilateral_filtered_frame = cv2.bilateralFilter(frame, 9, 75, 75)

    # دمج الصورة الأصلية والصورة بعد الترشيح لعرضهما جنبًا إلى جنب
    combined = cv2.hconcat([frame, bilateral_filtered_frame])

    # تغيير حجم الصورة المدمجة لتناسب النافذة المطلوبة
    resized_combined = cv2.resize(combined, (1400, 700))

    # عرض الإطارات المدمجة
    cv2.imshow('Original and Bilateral Filtered Frames', resized_combined)

    # الضغط على 'q' للخروج
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# إغلاق كل شيء
cap.release()
cv2.destroyAllWindows()
