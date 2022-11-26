def generate_faces(face_img, img_size=48):
    #Augment the detected face
    import cv2
    import numpy as np
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img[:, :])
    resized_images.append(face_img[2:45, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))
    resized_images.append(face_img[0:45, 0:45])
    resized_images.append(face_img[2:47, 0:45])
    resized_images.append(face_img[2:47, 2:47])

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images


def predict_data_gen():
    #Augmented prediction
    from model import CNN
    from data import Jaffe
    import numpy as np
    model = CNN()
    model.load_weights('../models/cnn_best_weights.h5')
    expression, x_test, y_test = Jaffe().gen_train()
    pred = []
    x_test = np.squeeze(x_test, axis=-1)
    for index in range(x_test.shape[0]):
        faces = generate_faces(x_test[index])
        results = model.predict(faces)
        result_sum = np.sum(results, axis=0)
        label_index = np.argmax(result_sum, axis=0)
        pred.append(label_index)
    pred = np.array(pred)
    print(np.sum(pred.reshape(-1) == y_test.reshape(-1)))


def predict_no_gen():
    #No augmented prediction
    from model import CNN
    from data import Jaffe
    import numpy as np
    model = CNN()
    model.load_weights('../models/cnn_best_weights.h5')
    expression, x_test, y_test = Jaffe().gen_train()
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    print(np.sum(pred.reshape(-1) == y_test.reshape(-1)))


if __name__ == '__main__':
    predict_data_gen()