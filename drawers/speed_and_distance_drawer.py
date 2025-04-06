import cv2

class SpeedAndDistanceDrawer():
    def __init__(self):
        pass 
    
    def draw_frame(self,frame,ball_aquisistion_player,player_distance,player_speed):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        font_scale = 0.7
        font_thickness=2

        # Overlay Position
        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * 0.01) 
        rect_y1 = int(frame_height * 0.67)
        rect_x2 = int(frame_width * 0.13)  
        rect_y2 = int(frame_height * 0.90)
        # Text positions
        text_x = int(frame_width * 0.03)  
        text_y0 = int(frame_height * 0.72) 
        text_y1 = int(frame_height * 0.80)  
        text_y2 = int(frame_height * 0.88)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255,255,255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        if ball_aquisistion_player >0:
            cv2.putText(
                frame, 
                f"Player: {ball_aquisistion_player}",
                (text_x, text_y0), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0,0,0), 
                font_thickness
            )


        if player_distance is not None:
            cv2.putText(
                frame, 
                f"{player_distance:.1f} M",
                (text_x, text_y1), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0,0,0), 
                font_thickness
            )

        if player_speed is not None:
            cv2.putText(
                frame, 
                f"{player_speed:.1f} Km/h",
                (text_x, text_y2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0,0,0), 
                font_thickness
            )


        return frame


    def draw(self, video_frames,ball_aquisition,player_distances_per_frame,player_speed_per_frame):
        output_video_frames = []
        previous_player_id = -1
        
        total_distances = {}

        for frame,frame_ball_aquisition,player_distance,player_speed in zip(video_frames,ball_aquisition,player_distances_per_frame,player_speed_per_frame):            
            default_player_id = -1 
            
            players_with_distances = list(player_distance.keys())
            if len(players_with_distances)>0:
                default_player_id = players_with_distances[0]

            if previous_player_id!=-1 and previous_player_id in player_distance:
                default_player_id = previous_player_id
            
            chosen_player_id=default_player_id

            if frame_ball_aquisition in player_distance:
                chosen_player_id = frame_ball_aquisition

            # Get Total Distance
            for player_id, distance in player_distance.items():
                if player_id not in total_distances:
                    total_distances[player_id]=0
                total_distances[player_id]+=distance


            chosen_player_distance = total_distances.get(chosen_player_id,None)
            chosen_player_speed = player_speed.get(chosen_player_id,None)
            
            previous_player_id = chosen_player_id

            frame_drawn = self.draw_frame(frame,chosen_player_id,chosen_player_distance, chosen_player_speed)
            output_video_frames.append(frame_drawn)

        return output_video_frames