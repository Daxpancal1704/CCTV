from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.to("cuda")

results = model("bus.jpg")

print(next(model.model.parameters()).device)