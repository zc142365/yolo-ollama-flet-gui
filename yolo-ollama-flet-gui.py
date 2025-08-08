import flet as ft
from ultralytics import YOLO
import cv2
import easyocr
import tempfile
import os
from PIL import Image
import io
import base64
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import requests
import threading
import concurrent.futures

# 전역 변수 선언
processing_message = None
file_picker = None
llm = None
page = None
model = None
reader = None
chat_history = []
chat_display = None
input_field = None
file_picker_result_text = None
file_selected = False
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
image_cache = {}

def main(page_param: ft.Page):
    global file_picker, llm, page, model, reader, chat_display, input_field, file_picker_result_text
    page = page_param
    page.title = "개선된 이미지 분석 및 채팅 앱"
    page.padding = 50
    page.theme_mode = "light"

    # 모델 초기화
    initialize_models()

    def show_processing_message():
        global processing_message
        processing_message = ft.Text("처리 중...")
        page.add(processing_message)
        page.update()

    def hide_processing_message():
        global processing_message
        if processing_message:
            page.controls.remove(processing_message)
            processing_message = None
            page.update()

    # Ollama 모델 목록 가져오기
    def get_ollama_models():
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            else:
                print(f"Error fetching models: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    ollama_models = get_ollama_models()

    # 모델 선택 드롭다운
    model_dropdown = ft.Dropdown(
        options=[ft.dropdown.Option(model) for model in ollama_models],
        width=200,
        label="Select Model",
    )

    # LangChain 설정
    def update_llm(e):
        global llm
        selected_model = model_dropdown.value
        # llm = Ollama(model=selected_model, base_url="http://172.26.32.1:11434")
        llm = Ollama(model=selected_model, base_url="http://localhost:11434")

    model_dropdown.on_change = update_llm

    image_analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 이미지 분석 전문가입니다. 주어진 이미지 정보를 바탕으로 상세하고 통찰력 있는 분석을 제공하세요. 사용자의 질문에 직접적으로 답변하고, 이미지의 맥락과 가능한 의미를 해석해주세요."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # 채팅 내역을 표시할 스크롤 가능한 컨테이너
    chat_display = ft.ListView(expand=True, spacing=10, auto_scroll=True)

    def stream_response(user_input, is_image_analysis=False):
        if llm is None:
            update_chat_history("System", "모델을 선택해주세요.")
            return

        if is_image_analysis:
            full_prompt = {
                "history": chat_history,
                "input": user_input
            }
            chain = image_analysis_prompt | llm | StrOutputParser()
        else:
            chain = llm | StrOutputParser()
        
        response = chain.invoke(full_prompt if is_image_analysis else user_input)
        update_chat_history("AI", response)
        return response

    def update_chat_history(sender, message, image=None):
        chat_history.append(f"{sender}: {message}")
        if image:
            chat_display.controls.append(ft.Column([
                ft.Text(f"{sender}:", weight=ft.FontWeight.BOLD),
                image,
                ft.Text(message, selectable=True)
            ]))
        else:
            chat_display.controls.append(ft.Text(f"{sender}: {message}", selectable=True))
        
        if len(chat_display.controls) > 10:
            chat_display.controls.pop(0)
        page.update()

    def reset_file_picker():
        global file_picker, file_selected
        file_picker = ft.FilePicker(on_result=pick_files_result)
        page.overlay.clear()
        page.overlay.append(file_picker)
        file_picker_result_text.value = ""
        file_selected = False
        page.update()

    def process_image(image_path):
        if image_path in image_cache:
            return image_cache[image_path]

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("이미지를 로드할 수 없습니다.")

        # 이미지 크기 조정
        image = cv2.resize(image, (640, 640))

        # YOLO 처리
        results = model(image)
        detected_objects = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy()
            names = result.names
            
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = box
                class_name = names[int(cls)]
                detected_objects.append(class_name)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # OCR 처리
        ocr_result = reader.readtext(image)
        detected_texts = [detection[1] for detection in ocr_result]

        detected_objects_str = ", ".join(detected_objects) if detected_objects else "객체 없음"
        detected_texts_str = ", ".join(detected_texts) if detected_texts else "텍스트 없음"

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='PNG')
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        image_preview = ft.Image(src_base64=img_base64, width=300, height=300, fit=ft.ImageFit.CONTAIN)

        image_cache[image_path] = (detected_objects_str, detected_texts_str, image_preview)
        return detected_objects_str, detected_texts_str, image_preview

    def process_input(e):
        global file_selected
        text_input = input_field.value
        
        show_processing_message()
        
        if text_input.startswith("/system "):
            # 시스템 명령 처리
            system_message = text_input[8:]  # "/system " 제거
            update_chat_history("System", f"시스템 메시지 설정: {system_message}")
            hide_processing_message()
            page.update()
            return
        elif file_selected and file_picker.result and file_picker.result.files:
            image_file = file_picker.result.files[0]
            try:
                detected_objects_str, detected_texts_str, image_preview = process_image(image_file.path)
                
                ollama_prompt = f"""
이미지 분석 결과:
- 감지된 객체: {detected_objects_str}
- 감지된 텍스트: {detected_texts_str}

사용자 질문: {text_input if text_input else "이미지에 대해 설명해주세요."}

위 정보를 바탕으로 다음 사항을 고려하여 응답해주세요:
1. 사용자의 질문에 직접적으로 답변하세요.
2. 이미지의 전반적인 구성과 분위기를 설명하세요.
3. 감지된 객체들 간의 관계나 상호작용을 추론해보세요.
4. 이미지에서 특별히 주목할 만한 요소나 특이점이 있다면 언급하세요.
5. 이미지가 전달하고자 하는 메시지나 의미에 대해 추측해보세요.
6. 필요하다면, 이미지와 관련된 추가적인 맥락이나 배경 정보를 제공하세요.

자연스럽고 대화체로 응답해주세요.
"""
                update_chat_history("User", text_input if text_input else "이미지에 대해 설명해주세요.", image_preview)
                threading.Thread(target=lambda: stream_response(ollama_prompt, is_image_analysis=True)).start()
            except Exception as e:
                print(f"이미지 처리 중 오류 발생: {e}")
                update_chat_history("System", f"이미지 처리 중 오류가 발생했습니다: {str(e)}")
        elif text_input:
            update_chat_history("User", text_input)
            threading.Thread(target=lambda: stream_response(text_input, is_image_analysis=False)).start()
        else:
            update_chat_history("System", "이미지나 텍스트를 입력해주세요.")
            hide_processing_message()
            return

        input_field.value = ""
        file_selected = False
        reset_file_picker()
        hide_processing_message()
        page.update()

    def on_send_button_click(e):
        process_input(e)

    def on_input_change(e):
        if e.control.value.endswith("\n"):
            e.control.value = e.control.value.rstrip()
            process_input(e)
            
    def pick_files_result(e: ft.FilePickerResultEvent):
        global file_selected
        if e.files:
            file_picker_result_text.value = ", ".join([f.name for f in e.files])
            file_selected = True
        else:
            file_picker_result_text.value = "파일 선택 취소됨."
            file_selected = False
        file_picker_result_text.update()
        page.update()

    file_picker = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(file_picker)

    select_button = ft.ElevatedButton("이미지 선택", on_click=lambda _: file_picker.pick_files(allow_multiple=False))

    input_field = ft.TextField(
        hint_text="텍스트를 입력하세요... (Enter로 전송)",
        expand=True,
        multiline=True,
        min_lines=1,
        max_lines=5,
        on_change=on_input_change
    )
    
    send_button = ft.ElevatedButton("전송", on_click=on_send_button_click)

    input_row = ft.Row([input_field, send_button])

    file_picker_result_text = ft.Text()

    page.add(
        ft.Column([
            ft.Row([select_button, model_dropdown]),
            chat_display,
            input_row,
            file_picker_result_text
        ], expand=True)
    )

    page.update()

def initialize_models():
    global model, reader
    model = YOLO('yolov8s.pt')
    reader = easyocr.Reader(['ko', 'en'])

if __name__ == "__main__":
    ft.app(target=main)