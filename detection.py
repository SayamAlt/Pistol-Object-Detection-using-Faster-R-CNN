import torch, torchvision, cv2, random
from collections import Counter
import numpy as np
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes=2):
    # Get the pretrained backbone model
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1")
    
    # Replace the classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def draw_bounding_boxes(image, boxes, labels, scores, class_names, color_sample, score_threshold=0.8):
    image_with_boxes = image.copy()
    
    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            xmin, ymin, xmax, ymax = map(int, box)
            
            color = random.choice(color_sample)
            
            # Draw the bounding box
            cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw the label
            label_text = f"{class_names[label]}: {score:.2f}"
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_width, text_height = text_size
            
            # Adjust the position of the label to prevent out-of-bounds
            ymin_text = max(ymin - text_height - 5, 0)
            
            # Background rectangle for the label text
            cv2.rectangle(image_with_boxes, (xmin, ymin_text - 5), (xmin + text_width + 10, ymin), color, -1)
            
            # Put text on the image with white color for better visibility
            cv2.putText(image_with_boxes, label_text, (xmin + 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
    return image_with_boxes
            
def pistol_detector(image, class_names, color_sample):
    # Load the model
    model = create_model(num_classes=len(class_names))
    model.load_state_dict(torch.load("pistol_object_detector.pth", map_location=torch.device("cpu")))
    model.eval() # Set the model to evaluation mode
    
    transform = transforms.ToTensor()
    
    # Convert the image to tensor
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0) # Add a batch dimension

    # Make a prediction on the image
    with torch.no_grad():
        prediction = model(image_tensor)[0] # Get the first image's predictions
        
    # Extract predictions
    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    
    # Convert image to Numpy array for drawing
    image = np.array(image)
    
    # Draw predictions on the image
    image_with_boxes = draw_bounding_boxes(image, boxes, labels, scores, class_names, color_sample)
    
    return image_with_boxes