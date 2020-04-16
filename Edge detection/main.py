import cv2
import edgeFilter

imgviewx=cv2.imread("data/img0.jpg",cv2.IMREAD_GRAYSCALE)
#imgviewx=cv2.imread("data/cat2.png")

edgeFilter.myEdgeFilter(imgviewx, 0.5)