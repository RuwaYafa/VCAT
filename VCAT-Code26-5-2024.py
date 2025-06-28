"""Thesis Title"""
# Final Project// VCAT/2024
## Ensure GPU Existance
!nvidia-smi
## Note: before start -> Prepare Paddleocr Packages from Processing OCR Request, because it needs restarting the system
! pip install paddleocr paddlepaddle
## Connect Colab to Drive
import sys
if 'google.colab' in sys.modules:
    %env GOOGLE_COLAB=1
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
else:
    %env GOOGLE_COLAB=0
    print("Warning: Not a Colab Environment")
## Extract Video Paths
# Preparing OS and Extracting videos Pathes
import os
import fnmatch

#  Getting a list of file names from nested folders after extracting digital forensic image
def find_videos(root_path):
    video_extensions = ['*.mp4', '*.mkv', '*.avi', '*.flv', '*.mov', '*.wmv', '*.mpeg', '*.ogv']
    video_files = []
    # Walking through the directory and its subdirectories
    for path, dirs, files in os.walk(root_path):
        for extension in video_extensions:
            for file in fnmatch.filter(files, extension):
                video_files.append(os.path.join(path, file))

    return video_files
## Get MetaData of Video File
# Preparing exiftool
!apt-get install exiftool
import subprocess
import json

def extract_video_metadata(video_path):
    """Run ExifTool command and return the output as JSON."""
    command = ['exiftool', '-json'] + [video_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.stdout:
        metadata = json.loads(result.stdout)
        if metadata:
            filename = video_path + 'metadata.json'
            with open(filename, 'w') as file:
                json.dump(metadata, file, indent=4) # indent=4 for pretty printing
            return metadata[0]

    else: return {}
## Prepering Helping Functions for Numerecal convergens and frame Combining
import numpy as np
import cv2

def time_converter(seconds):
    hours, seconds = np.divmod(seconds, 3600)
    minutes, seconds = np.divmod(seconds, 60)
    time_ = f'{hours:02.0f}:{minutes:02.0f}:{seconds:05.2f}'
    return time_

# Combine adjasent frames in one output intity, considering threshold value
def process_frames(frame_data, frame_threshold, frame_rate):
    output_frame = []
    for _ in frame_data.keys():
        first_frame = -1
        last_frame = -1
        cur = -1
        cur_frame = None
        current_confidence = 0
        for frame_id, confidence, frame in frame_data[_]:
            if first_frame == -1:
                first_frame = frame_id
                cur = frame_id
                cur_frame = frame
                cur_confidence = confidence
            else:
                if frame_id - cur > frame_threshold:
                    last_frame = cur
                    output_frame.append([_, time_converter(first_frame/frame_rate), time_converter(last_frame/frame_rate), cur_frame, cur_confidence])
                    first_frame = frame_id
                    cur_frame = frame
                    cur_confidence = confidence
                cur = frame_id
        if first_frame != -1:
            last_frame = cur
            output_frame.append([_, time_converter(first_frame/frame_rate), time_converter(last_frame/frame_rate), cur_frame, cur_confidence])
    return output_frame

import re
def convert_coordinates_gps(dms):
    parts = re.findall(r'\d+\.\d+|\d+', dms)
    degrees = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return degrees + minutes / 60 + seconds / 3600

import hashlib
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
## Processing OD Request
# Preparing Dino Model
HOME = os.getcwd()
%cd {HOME}

!git clone https://github.com/IDEA-Research/GroundingDINO.git
%cd {HOME}/GroundingDINO
!pip install -q -e .

CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")

# set model configuration file path
%cd {HOME}
!mkdir {HOME}/weights
%cd {HOME}/weights
!wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# set model weight file
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)

%cd {HOME}/GroundingDINO
from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
import groundingdino.util.inference
from PIL import Image
def get_object(dino_model, transform, predict, annotate, video_path, target_objects):
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    target_objects = ', '.join(target_objects)
    print('---------------------------------------',target_objects)
    detected_objects = {}
    frame_id = 0
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    while True:
        ret, frame = cap.read()
        if ret:
            image_transformed, _ = transform(Image.fromarray(frame).convert("RGB"), None) # PIL obj img to tensor
            boxes, logits, phrases = predict(
                model=dino_model,
                image=image_transformed,
                caption=target_objects,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                # device = 'cpu'   # Commet if run in GPU
            )
            annotated_frame = annotate(image_source=frame, boxes=boxes, logits=logits, phrases=phrases)
            %matplotlib inline
            # import supervision as sv
            # sv.plot_image(annotated_frame, (8, 8))
            for i in range(len(phrases)):
                phrase = phrases[i]
                if phrase not in detected_objects:
                    detected_objects[phrase] = []
                detected_objects[phrase].append([frame_id, float(logits[i]), annotated_frame])
            frame_id += 1
            if cv2.waitKey(1) == 27: 
                break
        else: break
    cap.release()
    return detected_objects
## Processing OCR Request
# Preparing PaddleOCR packages
# ! pip install paddleocr paddlepaddle
from paddleocr import PaddleOCR  # PaddleOCR for OCR tasks

def get_detected_ocr(ocr, frame_id, frame, target_objects, ocr_detected_objects):
    ocr_result = ocr.ocr(frame, cls=True)  # Perform OCR on the saved image # this line for paddler
    # ocr_result = ocr.readtext(frame, detail=1, paragraph=True) # this line for easyosr
    colors = np.random.uniform(0, 255, size=(100, 3))
    if ocr_result[0] is None:
        # print('none none')
        return ocr_detected_objects
    # frame_changed = False
    # Iterate through OCR results
    for i, line in enumerate(ocr_result[0]):  # this line for paddler
    # for i, (box, text, confidence) in enumerate(results):  # this line for easyosr
        box = line[0]       # this line for paddler
        text = line[1][0]   # this line for paddler
        conf = line[1][1]   # this line for paddler
        words = text.lower().strip() #.split()
        for target in target_objects:
            if target in words and conf>=0.1:
                # frame_changed = True
                # Draw the bounding box
                start_point = (int(box[0][0]), int(box[0][1]))  # Top-left corner
                end_point = (int(box[2][0]), int(box[2][1]))  # Bottom-right corner
                cv2.rectangle(frame, start_point, end_point, colors[i], 1)
                cv2.putText(frame, f'{target} ({conf:.2f})', (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
                if target not in ocr_detected_objects:
                    ocr_detected_objects[target] = []
                ocr_detected_objects[target].append([frame_id, conf, frame])
    return ocr_detected_objects

def  get_ocr(ocr, video_path, target_objects):
    ocr_detected_objects = {}
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if ret:
            ocr_detected_objects = get_detected_ocr(ocr, frame_id, frame, target_objects, ocr_detected_objects)

            if cv2.waitKey(1) == 27:  # ESC key to break
                break
        else: break
        frame_id += 1
    cap.release()
    return ocr_detected_objects
## Processing SR Requests
# base one for transcribting & searching exact word and phrases
!pip install openai-whisper
import whisper

def process_speech(wsr, option, video_path, target_objects):
    spoken_words = []
    detected_speech = {}
    confidence = 0
    start_time = 0
    end_time = 0
    spoken_words = wsr.transcribe(video_path, word_timestamps=True)
    for indx, segment in enumerate(spoken_words['segments']):
        segment['text'] = segment['text'].strip().lower()
        for target in target_objects:
            position_target = segment['text'].find(target)
            if(position_target !=-1):
                posistion_word = segment['text'].count(' ',0, position_target)
                sum_probability = 0
                count_probability = 0
                word = segment['words'][posistion_word + count_probability]
                start_time = time_converter(word['start'])
                for target_word in target.split():
                    # print('posistion_word + count_probability= ',posistion_word, count_probability)
                    word=segment['words'][posistion_word + count_probability]
                    sum_probability += word['probability']
                    count_probability +=1
                    end_time = time_converter(word['end'])
                if count_probability > 0:
                    confidence = sum_probability / count_probability
                    token = segment['text'].strip().lower()
                    if target not in detected_speech:
                        detected_speech[target] = []
                    detected_speech[target].append([indx+1, start_time, end_time, confidence, token]) # indx= segment No.
                else:
                    confidence = 0
    return detected_speech
## The Output Report Generation
!pip install reportlab
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch, cm  # Import inch for image sizing
import io

# Reporting .PDF
def make_pdf(configuration):
    pdf_path = project_configuration['report_path']+ "/Report.pdf"
    investigation_info = configuration['investigation_info']
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []
    # styles
    styles = getSampleStyleSheet()
    video_path_style = ParagraphStyle(
        'VideoPath',  # Name of the style
        parent=styles['Normal'],  # Base style
        textColor='blue',  # Text color (e.g., red)
        fontName='Helvetica-Bold',
        fontSize=10
    )
    sub_title = ParagraphStyle(
        'sub_title',  # Name of the style
        parent=styles['Normal'],  # Base style
        fontName='Helvetica-Bold',
        fontSize=12,
    )
    sub_title_result = ParagraphStyle(
        'sub_title_result',  # Name of the style
        parent=styles['Normal'],  # Base style
        fontName='Helvetica',
        fontSize=10,
        spaceBefore=6,
        spaceAfter=6,
        )
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),  # Header background color
        ('BACKGROUND', (0, 5), (-1, 5), colors.lightblue),  # Header background color
        ('BACKGROUND', (0, 12), (-1, 12), colors.lightblue),  # Header background color
        ('BACKGROUND', (0, 17), (-1, 17), colors.lightblue),  # Header background color
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Center align vertically
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Header font
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica'),  # Subheader font
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 0, colors.black),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ])
    table_style_result = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Center align vertically
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), # 'tradbdo'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0, colors.black),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ])
    # Adding a title
    title = "FORENSIC UNIT’S HEADER"
    title_style = styles['Title']  # Default title style, you can customize it as needed
    title_paragraph = Paragraph(title, style=title_style)
    spacer = Spacer(1, 0.25 * inch)  # Adds some space after the title
    # Creating a table for constant info
    analyst_name = investigation_info['analyst_name']
    analyst_organization = investigation_info['analyst_organization']
    analyst_department = investigation_info['analyst_department']
    sys_date = investigation_info['sys_date']
    duration = investigation_info['duration']
    case_number = investigation_info['case_number']
    report_id = investigation_info['report_id']
    start_date = investigation_info['start_date']
    end_date= investigation_info['end_date']
    Forensic_Image_Path= investigation_info['Forensic_Image_Path']
    Forensic_Image_size= investigation_info['Forensic_Image_size']
    MD5_Hash_Value = investigation_info['MD5_Hash_Value']
    data = [
        ['Video Analyst Information'],
        [f'Analyst Name: {analyst_name}'],
        [f'Analyst Organization: {analyst_organization}'],
        [f'Analyst Oepartment: {analyst_department}'],
        [],
        [Paragraph('Case Information', style=sub_title)],
        [f'Case Number: {case_number}'],
        [f'Report Number: {report_id}'],
        [f'Date: {sys_date}'],
        [],
        [Paragraph('Video Content Analysis Report', style=title_style)],
        [],
        [Paragraph('Work Time', style=sub_title)],
        [f'Start Date: {start_date}'],
        [f'End Date: {end_date}'],
        [f'Duration: {time_converter(duration)}'],
        [],
        [Paragraph('Forensic Image Information', style=sub_title)],
        [f'Forensic Image Path: {Forensic_Image_Path}'],
        [f'Forensic Image Size: {Forensic_Image_size}'],
        [f'MD5 Hash Value: {MD5_Hash_Value}'],
        [f'Note: \n\n']
    ]
    data_ocr = []
    data_object = []
    data_speech = []
    if configuration['object']:
        print('---------------------------',configuration['result_object'])
        data_object.append([Paragraph('Parsing Result of Object Detection', style=sub_title)])
        for video in configuration['result_object']:
            data_object.append([Paragraph('Video Path: '+ video[0], style=video_path_style)]) ###########################################
            file_md5 = md5(video[0])
            # extracted video information
            metadata = extract_video_metadata(video[0])
            imgwidth = metadata.get('ImageWidth')
            imgheight = metadata.get('ImageHeight')
            FileName = metadata.get('FileName')
            FileType = metadata.get('FileType')#
            FileSize = metadata.get('FileSize')
            FrameRate = metadata.get('VideoFrameRate')#
            FrameSize = f"{imgwidth}x{imgheight}"#
            Duration = metadata.get('Duration')#
            FileCreateDate = metadata.get('FileCreateDate')
            FileAccessDate = metadata.get('FileAccessDate')
            ModificationDate = metadata.get('ModifyDate')#
            DeviceMake = metadata.get('Make')#
            DeviceModel = metadata.get('Model')#
            # extracted GPS video information
            GPSPosition = metadata.get('GPSPosition')
            if GPSPosition is not None:
                latitude = metadata.get('GPSLatitude')
                longitude = metadata.get('GPSLongitude')
                latitude_decimal = convert_coordinates_gps(latitude)
                longitude_decimal = convert_coordinates_gps(longitude)
                google_maps_link = f"\"https://www.google.com/maps?q={latitude_decimal},{longitude_decimal}\""
            else:
                google_maps_link = "-"
            data_object.append([Paragraph('Video Information', style=sub_title)])
            data_object.append([
                '\nVideo Name: '+ str(FileName) +
                '\nFile Type: '+ str(FileType) + #
                '\nVideo File Size: '+ str(FileSize) +
                '\nMD5 Hash Value: '+ str(file_md5) +
                '\nFrame Rate: '+ str(FrameRate) + #
                '\nFrame Size: '+ str(FrameSize) + #
                '\nDuration: '+ str(Duration) + #
                '\nCreation Time: '+ str(FileCreateDate) +
                '\nLast Access Time: '+ str(FileAccessDate) +
                '\nModification Date: '+ str(ModificationDate) + #
                '\nDevice Make: '+ str(DeviceMake) + #
                '\nDevice Model: '+ str(DeviceModel) + #
                '\nLocation: '+ str(GPSPosition) +
                '\nGoogle Map Link: ' + str(google_maps_link) +
                '\n'
            ])
            ratio = imgwidth/ imgheight
            newwidth = 4
            newheigt = newwidth * ratio
            if newheigt > 6:
                newheigt = 6
                newwidth = newheigt /ratio
            for object in video[1]:
                # Convert NumPy array to PIL Image
                pil_img = Image.fromarray(object[3])
                # from IPython.display import display
                # display(pil_img)
                """ Convert PIL Image to bytes, allows you to work with the image data in-memory without having to save it to a file on disk."""
                img_bytes = io.BytesIO()
                pil_img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                # img = RLImage(img_bytes, width=(imgwidth/300)*inch, height=(imgheight/300)*inch)
                img = RLImage(img_bytes, width=newwidth*inch, height=newheigt*inch)
                print(object)
                data_object.append(['Keyword: '+ str(object[0]) + '\nTimestamp: \nStart: '+ object[1]+ ' - End: '+ object[2]+'\nConfidence: '+ str(object[4])])
                data_object.append([img])
    if configuration['ocr']:
        data_ocr.append([Paragraph('Parsing Result of Optical Character Recognition', style=sub_title)])
        for video in configuration['result_ocr']:
            data_ocr.append([Paragraph('Video Path: '+ video[0], style=video_path_style)]) 
            file_md5 = md5(video[0])
            # extracted video information
            metadata = extract_video_metadata(video[0])
            imgwidth = metadata.get('ImageWidth')
            imgheight = metadata.get('ImageHeight')
            FileName = metadata.get('FileName')
            FileType = metadata.get('FileType')#
            FileSize = metadata.get('FileSize')
            FrameRate = metadata.get('VideoFrameRate')#
            FrameSize = f"{metadata.get('ImageWidth')}x{metadata.get('ImageHeight')}"#
            Duration = metadata.get('Duration')#
            FileCreateDate = metadata.get('FileCreateDate')
            FileAccessDate = metadata.get('FileAccessDate')
            ModificationDate = metadata.get('ModifyDate')#
            DeviceMake = metadata.get('Make')#
            DeviceModel = metadata.get('Model')#
            # extracted GPS video information
            GPSPosition = metadata.get('GPSPosition')
            if GPSPosition is not None:
                latitude = metadata.get('GPSLatitude')
                longitude = metadata.get('GPSLongitude')
                latitude_decimal = convert_coordinates_gps(latitude)
                longitude_decimal = convert_coordinates_gps(longitude)
                google_maps_link = f"\"https://www.google.com/maps?q={latitude_decimal},{longitude_decimal}\""
            else:
                google_maps_link = "-"
            data_ocr.append([Paragraph('Video Information', style=sub_title)])
            data_ocr.append([
                '\nVideo Name: '+ str(FileName) +
                '\nFile Type: '+ str(FileType) + #
                '\nVideo File Size: '+ str(FileSize) +
                '\nMD5 Hash Value: '+ str(file_md5) +
                '\nFrame Rate: '+ str(FrameRate) + #
                '\nFrame Size: '+ str(FrameSize) + #
                '\nDuration: '+ str(Duration) + #
                '\nCreation Time: '+ str(FileCreateDate) +
                '\nLast Access Time: '+ str(FileAccessDate) +
                '\nModification Date: '+ str(ModificationDate) + #
                '\nDevice Make: '+ str(DeviceMake) + #
                '\nDevice Model: '+ str(DeviceModel) + #
                '\nLocation: '+ str(GPSPosition) +
                '\nGoogle Map Link: ' + str(google_maps_link) +
                '\n'
            ])
            ratio = imgwidth/ imgheight
            newwidth = 4
            newheigt = newwidth * ratio
            if newheigt > 6:
                newheigt = 6
                newwidth = newheigt /ratio
            for ocr in video[1]:
                # Convert NumPy array to PIL Image
                pil_img = Image.fromarray(ocr[3])
                # from IPython.display import display
                # display(pil_img)
                """Convert PIL Image to bytes, allows you to work with the image data in-memory without having to save it to a file on disk."""
                img_bytes = io.BytesIO()
                pil_img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                # img = RLImage(img_bytes, width=(imgwidth/300)*inch, height=(imgheight/300)*inch)
                img = RLImage(img_bytes, width=newwidth*inch, height=newheigt*inch)
                data_ocr.append(['Keyword: '+ ocr[0] + '\nTimestamp: \nStart:'+ ocr[1]+' - End: '+ ocr[2] + '\nConfidence: '+ str(ocr[4])])
                data_ocr.append([img])
    if configuration['speech']:
        data_speech.append([Paragraph('Parsing Result of Speech Recognition', style=sub_title)])
        for video in configuration['result_speech']:
            data_speech.append([Paragraph('Video Path: '+ video[0], style=video_path_style)]) ###########################################
            file_md5 = md5(video[0])
            # extracted video information
            metadata = extract_video_metadata(video[0])
            FileName = metadata.get('FileName')
            FileType = metadata.get('FileType')#
            FileSize = metadata.get('FileSize')
            FrameRate = metadata.get('VideoFrameRate')#
            FrameSize = f"{metadata.get('ImageWidth')}x{metadata.get('ImageHeight')}"#
            Duration = metadata.get('Duration')#
            FileCreateDate = metadata.get('FileCreateDate')
            FileAccessDate = metadata.get('FileAccessDate')
            ModificationDate = metadata.get('ModifyDate')#
            DeviceMake = metadata.get('Make')#
            DeviceModel = metadata.get('Model')#
            # extracted GPS video information
            GPSPosition = metadata.get('GPSPosition')
            if GPSPosition is not None:
                latitude = metadata.get('GPSLatitude')
                longitude = metadata.get('GPSLongitude')
                latitude_decimal = convert_coordinates_gps(latitude)
                longitude_decimal = convert_coordinates_gps(longitude)
                google_maps_link = f"\"https://www.google.com/maps?q={latitude_decimal},{longitude_decimal}\""
            else:
                google_maps_link = "-"
            data_speech.append([Paragraph('Video Information', style=sub_title)])
            data_speech.append([
                '\nVideo Name: '+ str(FileName) +
                '\nFile Type: '+ str(FileType) + #
                '\nVideo File Size: '+ str(FileSize) +
                '\nMD5 Hash Value: '+ str(file_md5) +
                '\nFrame Rate: '+ str(FrameRate) + #
                '\nFrame Size: '+ str(FrameSize) + #
                '\nDuration: '+ str(Duration) + #
                '\nCreation Time: '+ str(FileCreateDate) +
                '\nLast Access Time: '+ str(FileAccessDate) +
                '\nModification Date: '+ str(ModificationDate) + #
                '\nDevice Make: '+ str(DeviceMake) + #
                '\nDevice Model: '+ str(DeviceModel) + #
                '\nLocation: '+ str(GPSPosition) +
                '\nGoogle Map Link: ' + str(google_maps_link) +
                '\n'
            ])
            for target in video[1].keys():
                data_speech.append(['keyword: '+ target])
                for speech in video[1][target]:
                    data_speech.append(['Timestamp: \nStart: '+ speech[1] + ' - End: '+ speech[2] + '\nConfidence: '+ str(speech[3]) + '\nThe Context: '+ speech[4] ])
    # Specify column widths
    colWidths = [6.5*inch] 
    # Generate rowHeights to match the number of rows
    row_height_for_header = 0.5 * inch
    row_height_for_other_rows = 1.4 * inch
    rowHeights = [row_height_for_header] + [row_height_for_other_rows] * (len(data) - 1)  # Adjust row heights
    # Create the table with specified dimensions
    table = Table(data)
    table.setStyle(table_style)
    # Building the PDF
    elements = [title_paragraph, spacer, table]
    if configuration['object']:
        table_object = Table(data_object)
        table_object.setStyle(table_style_result)
        elements.append(spacer)
        elements.append(table_object)
    if configuration['ocr']:
        table_ocr = Table(data_ocr)
        table_ocr.setStyle(table_style_result)
        elements.append(spacer)
        elements.append(table_ocr)
    if configuration['speech']:
        table_speech = Table(data_speech)
        table_speech.setStyle(table_style_result)
        elements.append(spacer)
        elements.append(table_speech)
    doc.build(elements)
    print("PDF generated successfully!")
## The Main Function that Process VCAT Requests
# from google.colab.patches import cv2_imshow
from datetime import datetime
import time

def main(configuration):
    # record start time--------------------------
    start_time = time.time()
    start_date = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    ocr = None
    dino_model = None
    transform = None
    wsr = None
    option = None
    if configuration['object']:
        # load Dino model
        dino_model = load_model(CONFIG_PATH, WEIGHTS_PATH, 'GPU') ##### comment GPU if CPU 
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    if configuration['ocr']:
        # Initialize PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en')#, use_gpu=True)  # Assuming CPU usage
        # ocr = easyocr.Reader(['en'], gpu=False)  # Assuming CPU usage # this line for easyosr
    if configuration['speech']:
        # Initialize whisper
        wsr = whisper.load_model(
                name='base.en',
                # device= 'cpu', # comment it if GPU 
                in_memory= True
            )
        option = whisper.DecodingOptions(language='en', fp16=False)
    videos_paths = find_videos(configuration['root_directory'])
    result_ocr = []
    result_object = []
    result_speech = []
    print(videos_paths)
    for video_path in videos_paths:
        output_ocr = []
        output_object = []
        output_speech = []
        metadata = extract_video_metadata(video_path)
        frame_rate = metadata.get('VideoFrameRate')
        if configuration['object']:
            object_data = get_object(dino_model, transform, predict, annotate, video_path, configuration['object_val'])
            output_object = process_frames(object_data, configuration['frame_threshold'], frame_rate)
            if len(output_object) > 0:
                result_object.append([video_path, output_object])
                print('result_object', result_object)
        if configuration['ocr']:
            ocr_data = get_ocr(ocr, video_path, configuration['ocr_val'])
            output_ocr = process_frames(ocr_data, configuration['frame_threshold'], frame_rate)
            if len(output_ocr) > 0:
                result_ocr.append([video_path, output_ocr])
                print('result_ocr', result_ocr)
        if configuration['speech']:
            output_speech = process_speech(wsr, option, video_path, configuration['speech_val'])
            if len(output_speech) > 0:
                result_speech.append([video_path, output_speech])
                print('result_speech', result_speech)
    # take end time-----------------------
    end_time = time.time()
    end_date = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    # Calculate the execution time
    execution_time = end_time - start_time
    print("--------------------------------------- Execution time:", execution_time, "seconds")
    # write the pdf file
    if configuration['report']:
        investigation_info = configuration['investigation_info']
        investigation_info['start_date'] = start_date
        investigation_info['end_date'] = end_date
        investigation_info['duration'] = execution_time
        pdf_configuration = {
            'investigation_info': investigation_info,
            'ocr' : configuration['ocr'],
            'object' : configuration['object'],
            'speech' : configuration['speech'],
            'result_ocr' : result_ocr,
            'result_object' : result_object,
            'result_speech' : result_speech,
        }
        make_pdf(pdf_configuration)
## The Starting Point for Proposed Digital Forensics Video Contant Analysis Tool (VCAT)
#@markdown ###Digital Forensic Video Contant Analysis (VCAT)
#@markdown ---
#@markdown ####1. Video Analyst Information:

analyst_name = "Ruwa' Abu Hweidi" #@param {type:"string"}
analyst_organization = "PTUK" #@param {type:"string"}
analyst_department = 'Cybercrime_PTUK' #@param {type:"string"}
#@markdown ---

#@markdown ####2. Case Information:
case_number = '7102024' #@param {type:"string"}
report_id = "1948" #@param {type:"string"}
#@markdown ---

# # This step valid if the forensic image is extracted in the given path
# #@markdown ####Forensic Image Information:
# Forensic_Image_Path = "/content/drive/MyDrive/VCAT/project/data" #@param {type:"string"}
# #@markdown ---

#@markdown ####3. Input Path:
root_directory = '/content/drive/MyDrive/VCAT/project/data' #@param {type:"string"}
#@markdown ---

#@markdown ####4. Outpu Path:
report_path = "/content/drive/MyDrive/VCAT/project/result" #@param {type:"string"}
#@markdown ---

#@markdown #### 5. Keyword/Prompt:
# target_objects = 'Ruwa, MD-59, Post, Match direction of car, try it today, man, from, dear, user, Washington Bivd, 033-YCS, 59.legoland, list2023, step, parking, Final Shot, Original Plate, EACH PAGE, DOWNLOAD, depend on user restrictions, final mobile, mobile edit, mobile, data, hash, telegram, to continue with results, extraction, salamalicom, trim, function, important, soccer, ball, thats right, focus, fuck, tough guy, mother, mother fucker,man shadow, bag, walking man shoes, hat, yellow bike, wheel of black car, right buildings, umbrella' #@param {type:"string"}
target_objects = "solider, killer, kitchen, fight, Freaking awesome with that,  I'm going to chop off your goddamn head, dull knife, gangster, Shit, tough stuff, give you five bucks, goalie, best of the best, soccer player,  What's your name, soccer, ball, thats right, focus, fuck, tough guy, mother fucker" #@param {type:"string"}
target_objects = target_objects.split(',')
#@markdown ---
#@markdown #### 6. Search Choices:
Optical_Character_Recognition = False # @param {type:"boolean"}
Object_Detection = False # @param {type:"boolean"}
Speech_Recognition = False  # @param {type:"boolean"}
Genarate_Report = True  # @param {type:"boolean"}
#@markdown ---
#@markdown #### Analizing & Genarating the Report ...
# @title Click `Show code` in the code cell. { display-mode: "form" }

investigation_info = {
    # Video Analyst Information
    'analyst_name': analyst_name,
    'analyst_organization': analyst_organization,
    'analyst_department': analyst_department,
    # Case Information
    'case_number': case_number,
    'report_id': report_id,
    'sys_date': datetime.now().strftime("%d/%m/%Y"),
    # Work Time
    # 'start_date': datetime.now().strftime("%d/%m/%Y"),
    # 'end_date': datetime.now().strftime("%d/%m/%Y"),
    # Forensic Image Information
    'Forensic_Image_Path': root_directory,
    'Forensic_Image_size': '390.6GB',
    'MD5_Hash_Value': '7D1D5D54E9FE1E94CEEE88120BEC4A56'
}
project_configuration={
    'investigation_info': investigation_info,
    "root_directory":root_directory,
    'object': Object_Detection,
    'object_val': target_objects,
    'ocr': Optical_Character_Recognition,
    'ocr_val': target_objects,
    'speech': Speech_Recognition,
    'speech_val': target_objects,
    'report': Genarate_Report,
    'report_path': report_path,
    'frame_threshold': 15, # one frame in each 15 frame without any differance
    # 'conf_threshold': 0.5
   }
main(project_configuration)