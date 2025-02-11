import cv2
import numpy as np
import os

# فتح ملف الفيديو
cap = cv2.VideoCapture('020146-3173 (35).mp4')

# التحقق من فتح الفيديو بنجاح
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# إنشاء مجلد لحفظ الإطارات الفريدة
output_dir = 'unique_frames'
os.makedirs(output_dir, exist_ok=True)

# إعداد متغيرات للمقارنة
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame_gray = prev_frame_gray[:, 200:]
prev_frame_gray = cv2.equalizeHist(prev_frame_gray)

frame_count = 0
unique_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = frame_gray[:, 200:]
    # # تطبيق هيستوغرام التدرج اللوني
    frame_gray = cv2.equalizeHist(frame_gray)


    # حساب الفرق بين الإطار الحالي والإطار السابق
    frame_diff = cv2.absdiff(prev_frame_gray, frame_gray)
    mean_diff = np.mean(frame_diff)

    # إذا كان الفرق يتجاوز العتبة، يتم حفظ الإطار
    if mean_diff > 40:# يمكن تعديل العتبة حسب الحاجة
        unique_filename = os.path.join(output_dir, f'unique_frame_{unique_count}.jpg')
        cv2.imwrite(unique_filename, frame)
        unique_count += 1

    # تحديث الإطار السابق
    prev_frame_gray = frame_gray.copy()
    frame_count += 1

    # عرض الإطار الحالي والإطار الفريد الأخير
    combined = cv2.hconcat([prev_frame_gray, frame_gray])
    resized_combined = cv2.resize(combined, (1400, 700))
    cv2.imshow('Current and Previous Frames', resized_combined)

    # الضغط على 'q' للخروج
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# إغلاق كل شيء
cap.release()
cv2.destroyAllWindows()

print(f'Total frames processed: {frame_count}')
print(f'Unique frames saved: {unique_count}')
