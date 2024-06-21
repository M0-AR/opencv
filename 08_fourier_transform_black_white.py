import cv2
import numpy as np
from scipy.fftpack import fft2, fftshift

# فتح ملف الفيديو
cap = cv2.VideoCapture('(35).mp4')  # تأكد من تحديد مسار الفيديو هنا

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

    # تطبيق التحويل الفوريي
    fft_frame = fftshift(fft2(gray_frame))
    magnitude_spectrum = 20 * np.log(np.abs(fft_frame) + 1)  # استخدام +1 لتجنب القسمة على صفر

    # تحويل الطيف إلى صورة BGR للعرض المشترك
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    spectrum_image = cv2.convertScaleAbs(magnitude_spectrum)

    # دمج الصورة الأصلية وصورة الطيف الترددي
    combined = cv2.hconcat([cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR), cv2.cvtColor(spectrum_image, cv2.COLOR_GRAY2BGR)])

    # تغيير حجم الإطار المدمج ليناسب حجم النافذة المطلوب
    resized_combined = cv2.resize(combined, (1400, 700))  # عرض 800 وارتفاع 600

    # عرض الإطارات المدمجة
    cv2.imshow('Original and Fourier Transform Spectrum', resized_combined)

    # الضغط على 'q' للخروج
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# إغلاق كل شيء
cap.release()
cv2.destroyAllWindows()
