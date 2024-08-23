import gradio as gr
import cv2
import albumentations as A
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile

def process_file(file, text, file_type):
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_dir = os.path.join(desktop_path, "augmented_videos")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if file_type == "video":
        # Geçici dosya oluşturma
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(file)
            video_path = temp_file.name

        # Video dosyasını okuma
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Video dosyası açılamadı. Lütfen geçerli bir video dosyası yüklediğinizden emin olun."

        # Video özelliklerini alma
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(output_dir, "augmented_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Albumentations ile veri arttırma işlemleri
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),  # 30 dereceye kadar döndürme
            A.RandomBrightnessContrast(p=0.5)
        ])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            augmented_frame = transform(image=frame)['image']

            # Metni kareye ekleme
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(augmented_frame, text, (50, height - 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(augmented_frame)

        cap.release()
        out.release()

        if not os.path.isfile(output_path):
            return "Çıktı videosu oluşturulamadı. Lütfen tekrar deneyin."

        return output_path

    elif file_type == "image":
        # Geçici dosya oluşturma
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(file)
            image_path = temp_file.name

        # Resmi okuma
        image = Image.open(image_path)
        image = np.array(image)

        # Albumentations ile veri arttırma işlemleri
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),  # 30 dereceye kadar döndürme
            A.RandomBrightnessContrast(p=0.5)
        ])
        augmented_image = transform(image=image)['image']

        # Metni resme ekleme
        augmented_image = Image.fromarray(augmented_image)
        draw = ImageDraw.Draw(augmented_image)
        font = ImageFont.load_default()
        draw.text((50, augmented_image.height - 50), text, font=font, fill=(255, 255, 255))

        # Çıktı dosyasını oluşturma
        output_path = os.path.join(output_dir, "augmented_image.jpg")
        augmented_image.save(output_path)

        return output_path

# Gradio Arayüzü Oluşturma
interface = gr.Interface(
    fn=process_file,
    inputs=[
        gr.File(label="Dosya Seçin", type="binary"),  # binary olarak güncellendi
        gr.Textbox(label="Dosyaya Eklenecek Metin"),
        gr.Radio(["video", "image"], label="Dosya Türü")
    ],
    outputs="file",
    title="Metin Ekleme ve Veri Arttırma",
    description="Bu uygulama, video veya resme dinamik metin ekleyerek ve veri arttırımı uygulayarak yeni bir dosya oluşturur."
)

# Gradio Arayüzünü Başlatma
interface.launch(share=True)