from keras.utils import img_to_array,load_img



def get_jaffe_images():
    import cv2
    import os
    emotions = {
        'AN': 0,
        'DI': 1,
        'FE': 2,
        'HA': 3,
        'SA': 4,
        'SU': 5,
        'NE': 6
    }
    emotions_reverse = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']

    def detect_face(img):
        cascade = cv2.CascadeClassifier('../data/params/haarcascade_frontalface_alt.xml')
        rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            # no face has been detected
            return []
        rects[:, 2:] += rects[:, :2]
        return rects

    folder = '../data/jaffe'
    files = os.listdir(folder)
    images = []
    labels = []
    index = 0
    for file in files:
        img_path = os.path.join(folder, file)  # file path
        img_label = emotions[str(img_path.split('.')[-3][:2])]  # file name with label
        labels.append(img_label)
        img = cv2.imread(img_path, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects_ = detect_face(img_gray)
        for x1, y1, x2, y2 in rects_:
            cv2.rectangle(img, (x1+10, y1+20), (x2-10, y2), (0, 255, 255), 2)
            img_roi = img_gray[y1+20: y2, x1+10: x2-10]
            img_roi = cv2.resize(img_roi, (48, 48))
            images.append(img_roi)



        index += 1
    if not os.path.exists('../data/jaffe/Training'):
        os.mkdir('../data/jaffe/Training')
    for i in range(len(images)):
        path_emotion = '../data/jaffe/Training/{}'.format(emotions_reverse[labels[i]])
        if not os.path.exists(path_emotion):
            os.mkdir(path_emotion)
        cv2.imwrite(os.path.join(path_emotion, '{}.jpg'.format(i)), images[i])
    print("load jaffe dataset")


def expression_analysis(distribution_possibility):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    # define 8 facial expressions
    emotions = {
        '0': 'anger',
        '1': 'disgust',
        '2': 'fear',
        '3': 'happy',
        '4': 'sad',
        '5': 'surprised',
        '6': 'neutral',
        '7': 'contempt'
    }
    y_position = np.arange(len(emotions))
    plt.figure()
    #display the histogram
    plt.bar(y_position, distribution_possibility, align='center', alpha=0.5)
    plt.xticks(y_position, list(emotions.values()))
    plt.ylabel('possibility')
    plt.title('predict result')
    if not os.path.exists('../results'):
        os.mkdir('../results')
    plt.show()


def load_test_image(path):
    img = load_img(path, target_size=(48, 48), color_mode="grayscale")
    img = img_to_array(img) / 255.
    return img


def index2emotion(index):
    #Return a list of emotions according to the index of emotions
    emotions = {
        '0': 'anger',
        '1': 'disgust',
        '2': 'fear',
        '3': 'happy',
        '4': 'sad',
        '5': 'surprised',
        '6': 'neutral',
        '7': 'contempt'
    }
    return list(emotions.values())[index]

def cv2_img_add_text(img, text, left, top, text_color=(0, 255, 0)):
    #Add the information of emotions on the image
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text((left, top), text, text_color)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)







