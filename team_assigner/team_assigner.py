import numpy as np
import torch
from PIL import Image
import cv2
import os
import pickle
from transformers import AutoModel, AutoProcessor

import sys 
sys.path.append('../')
from utils import read_stub, save_stub

class TeamAssigner:
    def __init__(self,
                 team_1_class_name="white T-shirt",
                 team_2_class_name="blue T-shirt"):
        self.team_colors = {}
        self.player_team_dict = {}        
    
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

    def load_model(self):
        self.model = AutoModel.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2),:]

        # Convert to PIL Image
        rgb_image = cv2.cvtColor(top_half_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        image = [pil_image]

        
        text = [self.team_1_class_name, self.team_2_class_name]
        processed = self.processor(text=text, images=image, padding='max_length', return_tensors="pt")

        with torch.no_grad():
            image_features = self.model.get_image_features(processed['pixel_values'], normalize=True)
            text_features = self.model.get_text_features(processed['input_ids'], normalize=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        #argmax
        argmax_index = np.argmax(text_probs)
        
        return text[argmax_index]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
           return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id=2
        if player_color=="white T-shirt":
            team_id=1

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_player_teams_across_frames(self,video_frames,player_tracks,read_from_stub=False, stub_path=None):
        player_assignment = read_stub(read_from_stub,stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        self.load_model()

        player_assignment=[]
        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})
            for player_id, track in player_track.items():
                team = self.get_player_team(video_frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
                player_assignment[frame_num][player_id] = team
        
        save_stub(stub_path,player_assignment)

        return player_assignment