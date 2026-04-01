import numpy as np
from abc import ABC, abstractmethod

class BaseSportLogic(ABC):
    def __init__(self,config):
        self.config=config

    @abstractmethod
    def filter_ball(self,ball_detections,last_good_ball_box,frame_num,last_good_ball_frame):
        pass

class FootballLogic(BaseSportLogic):
    def filter_ball(self, ball_detections, last_good_ball_box, frame_num, last_good_ball_frame):
        best_idx=np.argmax(ball_detections.confidence)
        curr_box=ball_detections.xyxy[best_idx].tolist()

        return curr_box
    
class BasketballLogic(BaseSportLogic):
    def filter_ball(self, ball_detections, last_good_ball_box, frame_num, last_good_ball_frame):
        if last_good_ball_box is not None:
            prev_center=np.array([(last_good_ball_box[0]+last_good_ball_box[2])/2,
                                  (last_good_ball_box[1]+last_good_ball_box[3])/2])
            
            dists=[]
            for b in ball_detections.xyxy:
                curr_center=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])
                dists.append(np.linalg.norm(prev_center-curr_center))

            best_idx=np.argmin(dists)
        else:
            best_idx=np.argmax(ball_detections.confidence)

        return ball_detections.xyxy[best_idx].tolist()
    
class Tennislogic(BaseSportLogic):
    def filter_ball(self, ball_detections, last_good_ball_box, frame_num, last_good_ball_frame):
        if last_good_ball_box is not None:
            prev_center=np.array([(last_good_ball_box[0]+last_good_ball_box[2])/2,
                                  (last_good_ball_box[1]+last_good_ball_box[3])/2])
            
            dists=[]
            for b in ball_detections.xyxy:
                curr_center=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])
                dists.append(np.linalg.norm(prev_center-curr_center))

            best_idx=np.argmin(dists)
        else:
            best_idx=np.argmax(ball_detections.confidence)

        return ball_detections.xyxy[best_idx].tolist()
    
class SportLogicFactory:
    @staticmethod
    def get_logic(sport,config):
        logics = {
            'football': FootballLogic,
            'basketball': BasketballLogic,
            'tennis': Tennislogic
        }
        return logics.get(sport, FootballLogic)(config)