from dotenv import load_dotenv
import os
import urllib.request
import sys
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# Test Image URL
TEST_URL = 'https://cdn.smartrecruiters.com/blog/wp-content/uploads/2020/06/how-to-conduct-a-virtual-job-interview-740x494.jpg'

def main():
    try:
        # Load environment variables
        load_dotenv()

        endpoint = os.getenv("AI_SERVICE_ENDPOINT")
        key = os.getenv("AI_SERVICE_KEY")

        if not endpoint or not key:
            raise ValueError("Missing AI_SERVICE_ENDPOINT or AI_SERVICE_KEY environment variables.")

        client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

        # Analyze image
        result = client.analyze_from_url(
            image_url=TEST_URL,
            visual_features=[VisualFeatures.CAPTION, VisualFeatures.DENSE_CAPTIONS, VisualFeatures.TAGS, VisualFeatures.PEOPLE],
            gender_neutral_caption=True,
            language="en",
        )

        # Display results
        print(f'\nImage Caption: {result.caption.text.title()}')
        print(f'Confidence: {result.caption.confidence:.2f}\n')

        for index, tag in enumerate(result.tags.list):
            print(f"{index}: Tag: {tag.name} - Confidence: {tag.confidence:.2f}")

        print(f'People detected: {result.people.values if "values" in result.people else "No people detected"}')
        print(f'Dense captions: {result.dense_captions.values if "values" in result.dense_captions else "No dense captions available"}')

        draw_people(TEST_URL, result)

    except HttpResponseError as e:
        print(f"Azure API Error: {e}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

def draw_people(image_url, result):
    """ Draws bounding boxes around detected people in the image. """
    try:
        # Download and open image
        file_name = "test_image." + image_url.split('.')[-1]
        urllib.request.urlretrieve(image_url, file_name)
        image = Image.open(file_name)

        draw = ImageDraw.Draw(image)

        # Font setup
        try:
            font_path = "/Library/Fonts/Arial Unicode.ttf"
            font = ImageFont.truetype(font=font_path, size=12)
        except IOError:
            font = ImageFont.load_default()

        # Draw bounding boxes for detected people
        if "values" in result.people and result.people.values:
            for person in result.people.values:
                bbox = person.bounding_box
                x, y, width, height = bbox.x, bbox.y, bbox.w, bbox.h
                box_coords = (x, y, x + width, y + height)

                draw.rectangle(box_coords, outline="cyan", width=2)
                draw.text((x + 5, y - 15), "Person", font=font, fill="cyan")

        # Show the modified image
        image.show()

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
