from dotenv import load_dotenv
import os
import sys
import urllib.request
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

# Import Azure namespaces

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

testurl='https://cdn.britannica.com/68/216668-050-DD3A9D0A/United-States-President-Donald-Trump-2017.jpg'
testurl='https://149369349.v2.pressablecdn.com/wp-content/uploads/2017/01/interview.jpg'
#testurl='https://www.becomeopedia.com/wp-content/uploads//What-Is-A-Panel-Interview.jpg'
#testurl='https://cdn.uconnectlabs.com/wp-content/uploads/sites/33/2019/05/panel-interviews-480x250.jpg?v=21260'
#testurl='https://www.nbc.com/sites/nbcblog/files/styles/scale_862/public/2023/05/howie-mandel11.jpg'
#testurl='https://www.mediabistro.com/wp-content/uploads/2017/09/iStock-618179948.jpg'
testurl='https://cdn.smartrecruiters.com/blog/wp-content/uploads/2020/06/how-to-conduct-a-virtual-job-interview-740x494.jpg'
#testurl='https://koriburkholder.com/uploads/Fb1kvO1A/Virtualpanelinterviewpreparation.jpg'

def main():
    try:
        # Get Configuration Settings
        load_dotenv()

        client = ImageAnalysisClient(
            endpoint=os.environ["AI_SERVICE_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AI_SERVICE_KEY"])
        )       

        result = client.analyze_from_url(
        image_url=testurl,
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.DENSE_CAPTIONS,VisualFeatures.TAGS,VisualFeatures.people],
        gender_neutral_caption=True,
        language="en",
        )

        print(f'\nImage caption: {result.caption.text.title()}\nConfidence: {result.caption.confidence:0.2f}\n')
        for index, tags_dict in enumerate(result.tags.list):
            print(f"{index}:Tag: {tags_dict.name} -Confidence: {tags_dict.confidence:0.2f}")

        print(result.people['values'])
        print(result.dense_captions['values'])

        GetPeople(testurl,result)

    except Exception as ex:
        print(ex)

    
    # Display people results
def GetPeople(image_url,result):
    print('\n')

    # Open image file
    file_name='test'+'.'+image_url[image_url.rfind('.')+1:]
    urllib.request.urlretrieve(image_url,file_name)
    image = Image.open(file_name)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Prepare image for drawing
    fig = plt.figure(figsize=(800, 600))
    plt.axis('off')

    # Create a drawing object
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for facebox in result.dense_captions['values']:
        face_confidence=facebox['text']
        if face_confidence is not '':
            boundingBox=facebox['boundingBox']
        
            x,y,width,height=boundingBox['x'],boundingBox['y'],boundingBox['w'],boundingBox['h']
            box_coordinates = (x, y, x+width, y+height)

            # Draw a box (outline only) with a black color
            draw.rectangle(box_coordinates, outline="black", width=1)

            # Define the text and its position
            confidence_text = f'{face_confidence}'

            text_position = (x+10, y+10)  # 10px padding inside the box

            # Load a font (optional: you can use any .ttf font file you like)
            try:
                font_path='/Library/Fonts/Arial Unicode.ttf'
                font = ImageFont.truetype(font=font_path, size=10)  # Replace with a valid font path if needed
            except IOError:
                font = ImageFont.load_default()  # Fallback if the specific font is not available
                print('hello')

            # Add the text inside the box
            draw.text(text_position, confidence_text, font=font, fill="black")


    # Show the modified image
    image.show()

    return


if __name__ == "__main__":
    main()
