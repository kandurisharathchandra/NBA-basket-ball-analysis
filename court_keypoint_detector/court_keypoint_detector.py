from ultralytics import YOLO
import supervision as sv
import pickle 
import os
import sys 
sys.path.append('../')
from utils import read_stub, save_stub


class CourtKeypointDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def get_court_keypoints(self, frames,read_from_stub=False, stub_path=None):
        
        court_keypoints = read_stub(read_from_stub,stub_path)
        if court_keypoints is not None:
            if len(court_keypoints) == len(frames):
                return court_keypoints
        
        batch_size=20
        court_keypoints = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
            for detection in detections_batch:
                court_keypoints.append(detection.keypoints)

        save_stub(stub_path,court_keypoints)
        
        return court_keypoints
    
    def draw_court_keypoints(self, frames, court_keypoints):
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF1493'),
            radius=8)
        
        output_frames = []
        for index,frame in enumerate(frames):
            keypoints = court_keypoints[index]
            annotated_frame = frame.copy()
            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints)
            output_frames.append(annotated_frame)

        return output_frames