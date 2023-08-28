#스테이블 디퓨전 클론
#pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate scipy xformers

import PyPDF2
import folium
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import codecs
from os import listdir
from os.path import isfile, join
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import Document
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate)
import chardet
from langchain.text_splitter import TextSplitter
from typing import List
from flask import Flask, request, jsonify
from translate import Translator as TranslateLib
import os
from flask_cors import CORS
import re
from pytube import YouTube
import cv2
from PIL import Image
import clip
import torch
import math
#import numpy as np
import plotly.express as px
import datetime
import uuid

import tempfile
#from natural_language_youtube_search import VideoSearch  # Replace with the actual module name where VideoSearch is defined

import requests
import torch
import base64
import transformers
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
#from googleMap import googleMap



app = Flask(__name__)
CORS(app)
os.environ["OPENAI_API_KEY"] = "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"
chat_history = []
model_id = "stabilityai/stable-diffusion-2"
prompt = "a photo of an astronaut riding a horse on mars"
output_path = "astronaut_rides_horse.png"



class CharacterTextSplitter(TextSplitter):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        super().__init__(chunk_size, chunk_overlap)
        self._chunk_size = chunk_size

    def split_text(self, text: str) -> List[str]:
        result = []
        current_chunk = ""
        current_chunk_size = 0

        for char in text:
            if current_chunk_size + char_width(char) >= self._chunk_size:
                result.append(current_chunk)
                current_chunk = ""
                current_chunk_size = 0

            current_chunk += char
            current_chunk_size += char_width(char)

        if current_chunk:
            result.append(current_chunk)

        return result

    def split(self, doc: Document) -> List[Document]:
        result = []
        current_chunk = []
        current_chunk_size = 0

        for page in doc.pages:
            page_text = "".join(page)
            page_chunks = self.split_text(page_text)
            for chunk in page_chunks:
                current_chunk.append(chunk)
                current_chunk_size += sum(char_width(char) for char in chunk)
                if current_chunk_size >= self._chunk_size:
                    result.append(Document(pages=current_chunk))
                    current_chunk = []
                    current_chunk_size = 0

        if current_chunk:
            result.append(Document(pages=current_chunk))

        return result


class VideoSearch:
    def __init__(self, video_url):
        self.video_url = video_url
        self.N = 120
        self.download_video()
        self.extract_frames()
        self.load_clip_model()
        self.encode_frames()

    def download_video(self):
        streams = YouTube(self.video_url).streams.filter(adaptive=True, subtype="mp4", resolution="360p",
                                                         only_video=True)
        if len(streams) == 0:
            raise ValueError("No suitable stream found for this YouTube video!")
        print("Downloading...")
        streams[0].download(filename="video.mp4")
        print("Download completed.")

    def extract_frames(self):
        self.video_frames = []
        capture = cv2.VideoCapture('video.mp4')
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        current_frame = 0
        while capture.isOpened():
            ret, frame = capture.read()
            if ret == True:
                self.video_frames.append(Image.fromarray(frame[:, :, ::-1]))
            else:
                break
            current_frame += self.N
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        print(f"Frames extracted: {len(self.video_frames)}")

    def load_clip_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def encode_frames(self):
        batch_size = 256
        self.batches = math.ceil(len(self.video_frames) / batch_size)
        self.video_features = torch.empty([0, 512], dtype=torch.float16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for i in range(self.batches):
            batch_frames = self.video_frames[i * batch_size: (i + 1) * batch_size]
            batch_preprocessed = torch.stack([self.preprocess(frame) for frame in batch_frames]).to(device)
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_preprocessed)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)
            self.video_features = torch.cat((self.video_features, batch_features))
        print(f"Features: {self.video_features.shape}")

    def search_video(self, search_query, display_heatmap=True, display_results_count=1):
        with torch.no_grad():
            text_features = self.model.encode_text(clip.tokenize(search_query))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (100.0 * self.video_features @ text_features.T)
        values, best_photo_idx = similarities.topk(display_results_count, dim=0)
        if display_heatmap:
            print("Search query heatmap over the frames of the video:")
            fig = px.imshow(similarities.T.cpu().numpy(), height=50, aspect='auto', color_continuous_scale='viridis',
                            binary_string=True)
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            fig.show()
            print()

        result_images_paths = []
        for frame_id in best_photo_idx:
            frame_id = frame_id.item()
            result_image = self.video_frames[frame_id]
            image_path = self.save_image(result_image)
            result_images_paths.append(image_path)
            seconds = round(frame_id * self.N / self.fps)
            print(f"Found at {str(datetime.timedelta(seconds=seconds))} (Link: {self.video_url}&t={seconds})")

        return result_images_paths

    def save_image(self, image):
        # Define a directory where images will be stored within the Flask app's static folder
        image_dir = os.path.join(app.static_folder, 'images')
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f'image_{uuid.uuid4().hex}.png')
        image.save(image_path, format='PNG')
        return image_path


    # def handle_submit(video_url, query):
    #     result_images = search_video(video_url, query)
    #     # 이미지들을 HTML로 묶어서 스크롤 가능하도록 만듭니다.
    #     images_html = "<br>".join(
    #         [f"<img src='{img}' style='max-width: 100%; max-height: 300px; margin: 10px;'/>" for img in result_images])
    #     return images_html
    #
    # @staticmethod
    # def display_image(image):
    #     image.show()



def char_width(char: str) -> int:
    # 문자의 폭을 계산하는 함수 (한글은 2, 그 외는 1로 가정)
    return 2 if ord(char) > 127 else 1

def check_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# CSV 파일들의 인코딩 확인
def check_csv_encoding(csv_folder_path):
    for f in listdir(csv_folder_path):
        path = join(csv_folder_path, f)
        print("check_csv_encoding의 인코딩 확인 path :", path)
        if isfile(path) and '.csv' in f:
            encoding = check_encoding(path)
            print(f"{f}의 인코딩: {encoding}")
            if encoding.lower() == 'euc-kr':
                convert_csv_to_utf8(path, encoding)  # 변환 함수를 호출하여 인코딩이 'euc-kr'인 경우에만 변환 작업 수행





"""# CSV 파일 형식 변경"""
"""
euc-kr, cp949 파일을 읽어서 utf-8로 변환
utf-8 일때는 skip
file encoding
"""

def convert_csv_to_utf8(file_path, encoding):
    try:
        if encoding.lower() == 'euc-kr':
            with open(file_path, mode='r', encoding=encoding) as f:
                s = f.read()
            with open(file_path + '.bak', mode='w', encoding='euc-kr') as f:
                f.write(s)
            with open(file_path, mode='w', encoding='utf-8') as f:
                f.write(s)
            print(f"{file_path} 인코딩 변환 완료: {encoding} -> utf-8")
    except Exception as ex:
        print(f'[error] encoding : {encoding}, message : {ex}')
        return False
    else:
        return True


# 파일 인코딩을 변환하는 함수
def file_encoding_lst(dir_path):
    for f in listdir(dir_path):
        path = join(dir_path, f)
        print("file_encoding_lst의 path!! : ", path)
        if isfile(path) and '.csv' in f:
            encoding = check_encoding(path)
            print(f"{f}의 인코딩: {encoding}")
            if encoding.lower() == 'euc-kr':
                convert_csv_to_utf8(path, encoding)  # 인코딩이 euc-kr인 파일을 UTF-8로 변환
                # 추가된 부분: 변환한 후에 encoding을 다시 확인하고 출력
                encoding = check_encoding(path)
                print(f"{f}의 인코딩 변환 후: {encoding}")


def read_pdf(pdf_folder_path):
    pdf_texts = []
    pdf_files = [f for f in listdir(pdf_folder_path) if isfile(join(pdf_folder_path, f)) and f.endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = join(pdf_folder_path, pdf_file)
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
            pdf_texts.append(pdf_text)

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    pdf_texts_split = text_splitter.split_text(' '.join(pdf_texts))
    pdf_documents = [Document(page_content=content, metadata={'source': pdf_file}) for content in pdf_texts_split]
    return pdf_documents


def read_csv(csv_folder_path):
    csv_files = [f for f in listdir(csv_folder_path) if isfile(join(csv_folder_path, f)) and f.endswith('.csv')]
    texts = []
    for csv_file in csv_files:
        csv_path = join(csv_folder_path, csv_file)
        encoding = check_encoding(csv_path)
        print(f"{csv_file}의 인코딩: {encoding}")
        if encoding.lower() == 'euc-kr':
            convert_csv_to_utf8(csv_path, encoding)
        with codecs.open(csv_path, mode='r', encoding='utf-8') as f:
            csv_text = f.read()
            texts.append(csv_text)

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    csv_texts = text_splitter.split_text(''.join(texts))
    csv_documents = [Document(page_content=content, metadata={'source': csv_file}) for content in csv_texts]  # 'source' 메타데이터 추가
    return csv_documents


# respond 함수를 정의하여 채팅 응답 처리를 수행하는 부분을 아래와 같이 구현합니다.
def respond(message):
    result = chain(message)
    bot_message = result['answer'] + '\n'

    unique_sources = set()  # 중복을 방지하기 위한 set 선언

    for i, doc in enumerate(result['source_documents']):
        source = doc.metadata['source']
        if source not in unique_sources:  # 이미 해당 출처가 있는지 확인
            bot_message += ('출처: ' + source + '\n')
            unique_sources.add(source)  # 출처를 set에 추가

    chat_history.append((message, bot_message))  # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가

    return bot_message  # 수정된 채팅 기록을 반환


@app.route('/api/chat', methods=['GET', 'POST'])
def api_chat():
    data = request.json
    user_message = data.get('message')

    # 여기에서 챗봇 응답 처리 로직을 수행합니다.
    bot_message = respond(user_message)  # respond 함수를 사용하여 채팅 응답을 얻습니다.

    translated_bot_message = translate_message(bot_message)  # 번역 로직을 수행하여 번역된 메시지를 얻습니다.

    return jsonify({
        'user_message': user_message,
        'bot_message': translated_bot_message,  # 번역된 메시지로 변경
        'chat_history': chat_history
    })


# 챗봇 응답 처리
def chat_with_bot(user_message):
    result = chain({"question": user_message})
    bot_message = result['answer']
    chat_history.append((user_message, bot_message))

    # 번역 로직 추가
    translated_bot_message = translate_message(bot_message)

    return {
        'user_message': user_message,
        'bot_message': translated_bot_message,  # 번역된 메시지로 변경
        'chat_history': chat_history
    }

# 번역을 수행하는 함수
def translate_message(message, source_lang='ko', target_lang='ko'):
    translator = TranslateLib(to_lang=target_lang, from_lang=source_lang)
    translated_message = translator.translate(message)
    return translated_message


@app.route('/api/googlemap', methods=['POST'])
def show_map():
    response = chat_with_bot("경주의 latitude와 longitude를 알려줘") # chat_with_bot 함수 호출
    bot_response = response['bot_message'] # 'bot_message' 키에 해당하는 값 가져오기
    print("/api/googlemap^^bot_response",bot_response)
    latitude, longitude = input_hidden(bot_response)
    print("Received Latitude:", latitude)  # 반환된 값을 확인
    print("Received Longitude:", longitude)  # 반환된 값을 확인
    if latitude is not None and longitude is not None:
        location_map = folium.Map(location=[latitude, longitude], zoom_start=15)
        folium.Marker([latitude, longitude], popup='Your Location').add_to(location_map)
        return location_map._repr_html_()
    else:
        return "Latitude and/or Longitude values are missing."

    return location_map._repr_html_()


def input_hidden(bot_response):
    print("input_hidden^^bot_response", bot_response)

    # Remove parentheses only
    text = re.sub(r'[()]', '', bot_response)
    print("input_hidden^^text: ",text)

    latitude = None
    longitude = None

    latitude_match = re.search(r'latitude는 ([\d.]+)', text)
    longitude_match = re.search(r'longitude는 ([\d.]+)', text)

    if latitude_match:
        latitude_str = latitude_match.group(1)
        latitude = float(latitude_str)

    if longitude_match:
        longitude_str = longitude_match.group(1)
        longitude = float(longitude_str)

    print("Latitude:", latitude)
    print("Longitude:", longitude)

    return latitude, longitude




@app.route('/api/quiz', methods=['POST'])
@app.route('/api/recommend', methods=['POST'])
def process_user_message():
    data = request.json
    user_message = data.get('message')
    result = chat_with_bot(user_message)
    #print("result:",result)
    return jsonify(result)


@app.route('/api/youTube', methods=['POST'])
def search_youtube():
    data = request.json
    video_url = data.get('video_url')

    try:
        video_search = VideoSearch(video_url)
        result_images_paths = video_search.search_video(data.get('search_query'))

        response = {'success': True, 'result_images_paths': result_images_paths}
    except Exception as e:
        response = {'success': False, 'error_message': str(e)}

    return jsonify(response)


# @app.route('/api/diffusion', methods=['POST'])
# def generate_and_return_image():
#     data = request.json
#     model_id = data.get('model_id')
#     prompt = data.get('prompt')
#
#     # Use the Euler scheduler here instead
#     scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
#     pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16",
#                                                    torch_dtype=torch.float16)
#     pipe = pipe.to("cuda")
#
#     image = pipe(prompt, height=768, width=768).images[0]
#
#     # Convert image to base64
#     image_byte_array = image.to_byte_array(format='PNG')
#     image_base64 = base64.b64encode(image_byte_array).decode('utf-8')
#
#     return image_base64





file_encoding_lst('../aiProject/data')
check_csv_encoding('../aiProject/data') # CSV 파일들의 인코딩 확인
folder_path = '../aiProject/data'
texts = read_pdf(folder_path) + read_csv(folder_path)
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(texts, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
#generate_image_using_diffusion(model_id, prompt, output_path)

system_template="""
You are an AI chatbot specialized in Gyeongju, South Korea and Silla Dynasty in Korea.
There are two functions which are tourist attractions recommendation and a function to submit a quiz.
Answer as much as possible from the data you have learned.
Express external links as page titles.
Use the following pieces of context to answer the users question shortly.
Given the following summaries of a long document and a question.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.

[Recommend Tourist spots]
- You are an AI that recommends three tourist spots in Gyeongju, South Korea.
- You recommend Bulguksa Temple, Seokguram Grotto, and Cheomseongdae as priorities.

[Submit Quiz]
- Submit 5 questions.
- You should give the answer to the problem.
- Submit the quiz in a O,X format.


----------------
{summaries}
You MUST answer in Korean and in Markdown format:
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]

prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
css = """
#warning {background-color: #FFCCCB}
.feedback textarea {font-size: 20px !important;}
"""

if __name__ == '__main__':
    # RetrievalQAWithSourcesChain 인스턴스 생성
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    app.run(host='127.0.0.1', port=5000)
