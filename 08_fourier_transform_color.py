import cv2
import numpy as np
from scipy.fftpack import fft2, fftshift

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

    # إعداد مصفوفة لتخزين الطيف الترددي للصورة الملونة
    spectrum_images = []

    # تطبيق التحويل الفوريي على كل قناة لون
    for i in range(3):
        channel = frame[:, :, i]

        # تطبيق التحويل الفوريي
        fft_channel = fftshift(fft2(channel))
        magnitude_spectrum = 20 * np.log(np.abs(fft_channel) + 1)  # استخدام +1 لتجنب القسمة على صفر

        # تحويل الطيف إلى صورة BGR للعرض المشترك
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        spectrum_image = cv2.convertScaleAbs(magnitude_spectrum)
        spectrum_images.append(spectrum_image)

    # دمج صور الطيف للقنوات الملونة
    spectrum_combined = cv2.merge(spectrum_images)

    # دمج الصورة الأصلية وصورة الطيف الترددي
    combined = cv2.hconcat([frame, spectrum_combined])

    # تغيير حجم الإطار المدمج ليناسب حجم النافذة المطلوب
    resized_combined = cv2.resize(combined, (1400, 700))  # تعديل الأبعاد حسب الحاجة

    # عرض الإطارات المدمجة
    cv2.imshow('Original and Color Fourier Transform Spectrum', resized_combined)

    # الضغط على 'q' للخروج
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# إغلاق كل شيء
cap.release()
cv2.destroyAllWindows()
