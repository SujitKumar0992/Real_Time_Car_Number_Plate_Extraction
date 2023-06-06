import cv2
import easyocr
#import matplotlib.pyplot as plt
import cv2
import easyocr
#from IPython.display import Image

harcascade = "model/haarcascade_russian_plate_number.xml"
cap = cv2.VideoCapture(0)

cap.set(3, 640) # width
cap.set(4, 480) #height

min_area = 500
count = 0



def extract_text_from_image(image_path, language='en', output_path='output.txt'):
    # Initialize the EasyOCR reader with the desired language
    reader = easyocr.Reader([language])

    # Read the image and extract text
    result = reader.readtext(image_path)

    # Open the output file in write mode
    with open(output_path, 'w', encoding='utf-8') as file:
        # Write each line of extracted text to the file
        for entry in result:
            line = entry[1]
            file.write(line + '\n')

    print(f"Text extracted from the image has been saved to {output_path}")


while True:
    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x,y,w,h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y+h, x:x+w]
            cv2.imshow("ROI", img_roi)



    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)
        extract_text_from_image(img_roi)
        cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results",img)
        cv2.waitKey(500)
        count += 1

