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
    gray_frame = cv2.equalizeHist(gray_frame)

    # تحديد الحواف باستخدام Canny
    edges = cv2.Canny(gray_frame, 50, 150)

    # العثور على الكونتورز
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # إعداد صورة لرسم الكونتورز
    contour_img = np.zeros_like(gray_frame)

    # حساب مقاييس هو لكل كونتور ورسمها
    for contour in contours:
        moments = cv2.moments(contour)
        huMoments = cv2.HuMoments(moments)
        print("Hu Moments:", huMoments.flatten())  # طباعة مقاييس هو
        cv2.drawContours(contour_img, [contour], -1, 255, 2)  # رسم باللون الأبيض


    # دمج الصورة الأصلية وصورة الكونتورز لعرضهما جنبًا إلى جنب
    combined = cv2.hconcat([cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR), cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)])

    # تغيير حجم الصورة المدمجة لتناسب النافذة المطلوبة
    resized_combined = cv2.resize(combined, (1400, 700))

    # عرض الإطارات المدمجة
    cv2.imshow('Original and Contours with Hu Moments', resized_combined)

    # الضغط على 'q' للخروج
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# إغلاق كل شيء
cap.release()
cv2.destroyAllWindows()
