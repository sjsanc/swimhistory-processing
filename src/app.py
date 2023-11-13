import io
from dotenv import load_dotenv
import os

from flask import Flask, jsonify, request
from celery import Celery
import boto3
import botocore
import psycopg2
import pytesseract
import cv2
import requests
from PIL import Image
import face_recognition
from psycopg2.extras import Json

load_dotenv()

app = Flask(__name__)

app.config['CELERY_BROKER_URL'] = 'pyamqp://guest@localhost//'
app.config['CELERY_RESULT_BACKEND'] = 'rpc://'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

session = boto3.session.Session()

client = session.client(
    's3',
    endpoint_url=os.environ.get('SPACES_ENDPOINT'),
    config=botocore.config.Config(s3={'addressing_style': 'virtual'}),
    region_name=os.environ.get('SPACES_REGION'),
    aws_access_key_id=os.environ.get('SPACES_KEY'),
    aws_secret_access_key=os.environ.get('SPACES_SECRET')
)

@celery.task
def extract_text_from_image(id):
    try:
        response = client.list_objects_v2(
            Bucket=os.environ.get('SPACES_BUCKET'),
            Prefix=f"{str(id)}/original"
        )

        if 'Contents' in response:
            object_key = response['Contents'][0]['Key']
            client.download_file(os.environ.get('SPACES_BUCKET'), object_key, f'/tmp/{str(id)}')
            image = cv2.imread(f'/tmp/{str(id)}')

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
            dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)
            contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            image_copy = image.copy()

            extracted_text = ""

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cropped = image_copy[y : y + h, x : x + w]
                text = pytesseract.image_to_string(cropped)
                extracted_text += text + "\n"

            url = 'http://localhost:5050/api/images.updateImageText'
            json = { 'text': extracted_text, 'id': id}
            headers = {'Content-Type': 'application/json'}

            response = requests.post(url, headers=headers, json=json)

            return response.json()



    except Exception as e:
        print(f"Error: {e}")


@app.route('/ocr', methods=['POST'])
def handle_ocr():
    data = request.json
    image_id = data.get('imageId', 0)
    result = extract_text_from_image.delay(image_id)
    return jsonify({"task_id": result.id}), 202

@celery.task
def resize_image(id):
    try: 
        response = client.list_objects_v2(
            Bucket=os.environ.get('SPACES_BUCKET'),
            Prefix=f"{str(id)}/original"
        )

        if 'Contents' in response:
            object_key = response['Contents'][0]['Key']
            client.download_file(os.environ.get('SPACES_BUCKET'), object_key, f'/tmp/{str(id)}')
            image = Image.open(f'/tmp/{str(id)}')

            for size in [500, 100]:
                resized_image = image.copy()
                resized_image.thumbnail((size, size))
                buffer = io.BytesIO()
                resized_image.save(buffer, format='WEBP')
                
                client.put_object(
                    Bucket=os.environ.get('SPACES_BUCKET'),
                    Key= f'{str(id)}/{size}.webp',
                    Body=buffer.getvalue(),
                    ACL='public-read',
                    ContentType='image/webp'
                )


    except Exception as e:
        print(f"Error: {e}")

@app.route('/resize', methods=['POST'])
def handle_resize():
    data = request.json
    image_id = data.get('imageId', 0)
    result = resize_image.delay(image_id)
    return jsonify({"task_id": result.id}), 202


@celery.task
def detect_faces(id):
    try:
        response = client.list_objects_v2(
            Bucket=os.environ.get('SPACES_BUCKET'),
            Prefix=f"{str(id)}/original"
        )

        

        if 'Contents' in response:
            
            object_key = response['Contents'][0]['Key']
            client.download_file(os.environ.get('SPACES_BUCKET'), object_key, f'/tmp/{str(id)}')

            image = face_recognition.load_image_file(f'/tmp/{str(id)}')
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            face_encodings_list = [encoding.tolist() for encoding in face_encodings]

            # try:
            #     connection = psycopg2.connect(
            #         user="postgres",
            #         password="WuBK89m08a6",
            #         host="dtoelklrflwzvknijczc.supabase.co",
            #         port="5432",
            #         database="postgres"
            #     )

            #     cursor = connection.cursor()

            #     cursor.execute("""
            #         SELECT * FROM "Embedding";
            #     """)

            #     result = cursor.fetchall()

            #     print(result)
            #     connection.commit()
            #     cursor.close()
            #     connection.close()
            #     return(result)
                

            #     for encoding_list in face_encodings_list:
            #         # Convert the list to a JSONB object for pgvector
            #         encoding_json = Json(encoding_list)

            #         # Compare with existing Embeddings
            #         cursor.execute("""
            #             SELECT id, person_id, image_id
            #             FROM embedding
            #             WHERE embedding <-> %s <@ 0.1
            #             LIMIT 1;
            #         """, (encoding_json,))
                    
            #         result = cursor.fetchone()

            #         if result:
            #             # If a matching Embedding is found, associate it with the current image
            #             embedding_id, person_id, image_id = result
            #         else:
            #             # If no matching Embedding is found, create a new Person
            #             cursor.execute("""
            #                 INSERT INTO person DEFAULT VALUES
            #                 RETURNING id;
            #             """)
            #             person_id = cursor.fetchone()[0]

            #             # Insert a new Embedding
            #             cursor.execute("""
            #                 INSERT INTO embedding (embedding, person_id, image_id)
            #                 VALUES (%s, %s, %s)
            #                 RETURNING id;
            #             """, (encoding_json, person_id, image_id))
            #             embedding_id = cursor.fetchone()[0]

            #     # Commit the transaction
            #     connection.commit()

            # except psycopg2.Error as e:
            #     # Handle any database errors
            #     print(f"Database Error: {e}")
            #     connection.rollback()

            # finally:
            #     # Close the database connection
            #     cursor.close()
            #     connection.close()

    except Exception as e:
        # Handle any other errors
        print(f"Error: {e}")


@app.route("/face-detection", methods=['POST'])
def handle_face_detection():
    data = request.json
    image_id = data.get('imageId', 0)
    result = detect_faces.delay(image_id)
    return jsonify({"task_id": result.id}), 202

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)

