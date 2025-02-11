import cv2
import numpy as np

# فتح ملف الفيديو
cap = cv2.VideoCapture('020146-3173 (35).mp4')

# التحقق من فتح الفيديو بنجاح
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# قراءة الإطار الأول لاستخدامه كقالب
ret, first_frame = cap.read()
first_frame = first_frame[:, 200:]
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

# تحويل الإطار الأول إلى الأبيض والأسود لاستخدامه كقالب
template = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]

# قراءة الفيديو إطاراً تلو الآخر
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = frame[:, 200:]

    # تحويل الإطار إلى الأبيض والأسود
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # تطبيق مطابقة النمط (Template Matching)
    res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    # رسم مستطيل حول الأماكن التي تطابق النمط
    matched_frame = frame.copy()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(matched_frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    # دمج الصورة الأصلية والصورة بعد مطابقة النمط
    combined = cv2.hconcat([frame, matched_frame])

    # تغيير حجم الصورة المدمجة لتناسب النافذة المطلوبة
    resized_combined = cv2.resize(combined, (1400, 700))

    # عرض النتائج
    cv2.imshow('Original and Template Matching Results', resized_combined)

    # الضغط على 'q' للخروج
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# إغلاق كل شيء
cap.release()
cv2.destroyAllWindows()
