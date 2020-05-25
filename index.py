import cv2

from segmenter import segmentary, readImageAndResize
from nnparkinson import predict


labels = ("Control", "Patient")

def start():
    file_path = input("Digite o caminho da imagem: ")

    image = readImageAndResize(file_path)
    blueLineImage, _ = segmentary(image)

    predict_image = predict(blueLineImage)
        
    print("Predict:", labels[predict_image])


if __name__ == "__main__":
    start()
