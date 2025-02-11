import cv2
import numpy as np

# فتح ملف الفيديو
cap = cv2.VideoCapture('020146-3173 (35).mp4')  # تأكد من تحديد مسار الفيديو

# التحقق من فتح الفيديو بنجاح
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# قراءة الفيديو إطاراً تلو الآخر
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = frame[:, 200:]
    # تحويل الإطار إلى الأبيض والأسود
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # تطبيق هيستوغرام التدرج اللوني
    gray_frame = cv2.equalizeHist(gray_frame)

    # استخدام Canny لكشف الحواف
    edges = cv2.Canny(gray_frame, 50, 150, apertureSize=3)

    # تحويل edges إلى صورة ملونة لتوحيد نوع البيانات
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # استخدام تحويل هاف للكشف عن الخطوط
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    hough_frame = frame.copy()  # نسخ الإطار لرسم الخطوط عليه
    if lines is not None:
        for rho, theta in lines[:,0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(hough_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # دمج الصورة الأصلية (قبل الكشف) والصورة بعد تطبيق تحويل هاف
    combined = cv2.hconcat([cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR), edges_colored, hough_frame])

    # تغيير حجم الصورة المدمجة لتناسب النافذة المطلوبة
    resized_combined = cv2.resize(combined, (1400, 700))

    # عرض الإطارات المدمجة
    cv2.imshow('Original, Edges, and Hough Transform Lines', resized_combined)

    # الضغط على 'q' للخروج
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# إغلاق كل شيء
cap.release()
cv2.destroyAllWindows()
