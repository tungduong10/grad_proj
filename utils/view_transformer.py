import numpy as np
import cv2

class ViewTransformer:
    def __init__(self,sport,pixel_vertices,configs):
        self.sport=sport
        court_width, court_length=configs[sport]['court_dim']

        self.pixel_vertices=np.array(pixel_vertices).astype(np.float32)

        self.target_vertices=np.array ([
            [0,court_width],
            [0,0],
            [court_length,0],
            [court_length,court_width]
        ]).astype(np.float32)

        self.perspective_transformer=cv2.getPerspectiveTransform(
            self.pixel_vertices,
            self.target_vertices
        )

    def transform_point(self,point):
        p=(int(point[0]),int(point[1]))

        #Check if the player is inside the court lines
        is_inside=cv2.pointPolygonTest(self.pixel_vertices,p,False)
        if not is_inside:
            return None
        
        # Convert video pixels to real-world meters
        reshaped_point = np.array(point).reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        
        return transform_point.reshape(-1, 2).squeeze().tolist()