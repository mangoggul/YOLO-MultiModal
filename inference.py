import cv2
import time
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('new_data_yolov8nSOS/yolov8nSOS3/weights/best.pt')

image_paths = [ #임의의 연속적인 image 4장을 준비
    "dataset/new_data/test/RGB/000045.jpg",
    "dataset/new_data/test/RGB/000046.jpg",
    "dataset/new_data/test/RGB/000047.jpg",
    "dataset/new_data/test/RGB/000048.jpg",
    "dataset/new_data/test/RGB/000052.jpg",
    "dataset/new_data/test/RGB/000053.jpg",
    "dataset/new_data/test/RGB/000054.jpg",
    "dataset/new_data/test/RGB/000055.jpg",
    "dataset/new_data/test/RGB/000056.jpg",
    "dataset/new_data/test/RGB/000057.jpg",
    "dataset/new_data/test/RGB/000058.jpg",
    "dataset/new_data/test/RGB/000059.jpg",
    "dataset/new_data/test/RGB/000060.jpg"
]
depth_image_paths = [
    "dataset/new_data/test/D/000045.jpg",
    "dataset/new_data/test/D/000046.jpg",
    "dataset/new_data/test/D/000047.jpg",
    "dataset/new_data/test/D/000048.jpg",
    "dataset/new_data/test/D/000052.jpg",
    "dataset/new_data/test/D/000053.jpg",
    "dataset/new_data/test/D/000054.jpg",
    "dataset/new_data/test/D/000055.jpg",
    "dataset/new_data/test/D/000056.jpg",
    "dataset/new_data/test/D/000057.jpg",
    "dataset/new_data/test/D/000058.jpg",
    "dataset/new_data/test/D/000059.jpg",
    "dataset/new_data/test/D/000060.jpg"
]
thermo_image_paths = [
    "dataset/new_data/test/Thermo/000045.jpg",
    "dataset/new_data/test/Thermo/000046.jpg",
    "dataset/new_data/test/Thermo/000047.jpg",
    "dataset/new_data/test/Thermo/000048.jpg",
    "dataset/new_data/test/Thermo/000052.jpg",
    "dataset/new_data/test/Thermo/000053.jpg",
    "dataset/new_data/test/Thermo/000054.jpg",
    "dataset/new_data/test/Thermo/000055.jpg",
    "dataset/new_data/test/Thermo/000056.jpg",
    "dataset/new_data/test/Thermo/000057.jpg",
    "dataset/new_data/test/Thermo/000058.jpg",
    "dataset/new_data/test/Thermo/000059.jpg",
    "dataset/new_data/test/Thermo/000060.jpg"
]

total_inference_time = 0 
num_images = len(image_paths)

for i in range(num_images):
    
    img = cv2.imread(image_paths[i])
    img2 = cv2.imread(depth_image_paths[i])
    img3 = cv2.imread(thermo_image_paths[i])
    
    start_time = time.time()
    
    results = model(img, img2, img3)
    
    end_time = time.time()
    
    inference_time = end_time - start_time
    total_inference_time += inference_time

    # 결과 시각화
    for j, result in enumerate(results):
        if hasattr(result, 'boxes'):
            # 이미지에 바운딩 박스 그리기
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # 바운딩 박스의 좌표
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 바운딩 박스가 그려진 이미지 표시
            cv2.imshow(f"YOLOv8 추론 - 이미지 {i+1}", img)
        else:
            print(f"Result {j+1} does not have bounding boxes.")

# 평균 FPS 계산
average_inference_time = total_inference_time / num_images
fps = 1 / average_inference_time
print(f"Average Inference Time: {average_inference_time:.4f} seconds")
print(f"Average FPS: {fps:.2f}")

# 키 입력 대기 후 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
