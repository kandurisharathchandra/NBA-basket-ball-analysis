import cv2
import numpy as np

class PassInterceptionDrawer:
    def __init__(self):
        pass

    def get_stats(self, passes, interceptions):
        team1_passes = []
        team2_passes = []
        team1_interceptions = []
        team2_interceptions = []

        for frame_num, (pass_frame, interception_frame) in enumerate(zip(passes, interceptions)):
            if pass_frame == 1:
                team1_passes.append(frame_num)
            elif pass_frame == 2:
                team2_passes.append(frame_num)
                
            if interception_frame == 1:
                team1_interceptions.append(frame_num)
            elif interception_frame == 2:
                team2_interceptions.append(frame_num)
                
        return len(team1_passes), len(team2_passes), len(team1_interceptions), len(team2_interceptions)

    def draw(self, video_frames, passes, interceptions):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            if frame_num == 0:
                continue
            
            frame_drawn = self.draw_frame(frame, frame_num, passes, interceptions)
            output_video_frames.append(frame_drawn)
        return output_video_frames
    
    def draw_frame(self, frame, frame_num, passes, interceptions):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        
        # Overlay Position
        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * 0.60)
        rect_y1 = int(frame_height * 0.55)
        rect_x2 = int(frame_width * 0.99)
        rect_y2 = int(frame_height * 0.70)
        
        # Text positions
        text_x = int(frame_width * 0.63)
        text_y1 = int(frame_height * 0.60)
        text_y2 = int(frame_height * 0.65)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255,255,255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Get stats until current frame
        passes_till_frame = passes[:frame_num+1]
        interceptions_till_frame = interceptions[:frame_num+1]
        
        team1_passes, team2_passes, team1_interceptions, team2_interceptions = self.get_stats(
            passes_till_frame, 
            interceptions_till_frame
        )

        cv2.putText(
            frame, 
            f"Team 1 - Passes: {team1_passes} Interceptions: {team1_interceptions}",
            (text_x, text_y1), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0,0,0), 
            3
        )
        
        cv2.putText(
            frame, 
            f"Team 2 - Passes: {team2_passes} Interceptions: {team2_interceptions}",
            (text_x, text_y2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0,0,0), 
            3
        )

        return frame