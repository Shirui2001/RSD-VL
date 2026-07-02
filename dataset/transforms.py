"""
Data augmentation transforms for road anomaly detection.
RandomRoadObject: Synthesizes random polygon objects on road regions.
"""

import numpy as np
import random
from PIL import Image, ImageDraw
import cv2


class RandomRoadObject:
    """
    Random road object augmentation: synthesizes random polygon objects on road regions.
    
    Based on EUG's RandomRoadObject logic:
    - Generates random polygons on road regions (label=1)
    - Output mask: 0 (synthesized object/anomaly), 1 (normal road), 255 (ignore)
    
    Args:
        rcp: Random object placement probability (default: 0.5)
        max_random_poly: Maximum number of random polygons (default: 10)
        min_sz_px: Minimum polygon size in pixels (default: 32)
        max_sz_px: Maximum polygon size in pixels (default: 256)
    """
    
    def __init__(
        self,
        rcp: float = 0.5,
        max_random_poly: int = 10,
        min_sz_px: int = 32,
        max_sz_px: int = 256,
    ):
        self.rcp = rcp  # Random object placement probability
        self.max_random_poly = max_random_poly
        self.min_sz_px = min_sz_px
        self.max_sz_px = max_sz_px
    
    def __call__(self, sample):
        """
        Apply random road object augmentation.
        
        Args:
            sample: Dictionary with keys:
                - "image": PIL Image (RGB)
                - "label": PIL Image (L mode, road mask: 1=road, 255=ignore)
                - "randseed": Optional random seed for reproducibility
        
        Returns:
            Dictionary with keys:
                - "image": Augmented PIL Image (RGB)
                - "label": Updated mask (0=anomaly, 1=normal, 255=ignore)
        """
        image = sample["image"]
        label = sample["label"]
        randseed = sample.get("randseed", None)
        
        # Set random seed if provided
        if randseed is not None:
            random.seed(randseed)
            np.random.seed(randseed)
        
        # Convert to numpy arrays
        img_array = np.array(image)
        label_array = np.array(label)
        
        # Create output mask (copy of input label)
        output_mask = label_array.copy()
        
        # Decide whether to place random objects
        if random.random() < self.rcp:
            # Find road regions (label == 1)
            road_mask = (label_array == 1)
            
            if road_mask.any():
                # Get road region coordinates
                road_coords = np.where(road_mask)
                if len(road_coords[0]) > 0:
                    # Generate random number of polygons
                    num_poly = random.randint(1, self.max_random_poly)
                    
                    for _ in range(num_poly):
                        # Random polygon size
                        poly_size = random.randint(self.min_sz_px, self.max_sz_px)
                        
                        # Random center position within road region
                        if len(road_coords[0]) > 0:
                            idx = random.randint(0, len(road_coords[0]) - 1)
                            center_y = road_coords[0][idx]
                            center_x = road_coords[1][idx]
                            
                            # Generate random polygon vertices
                            num_vertices = random.randint(3, 8)  # 3-8 vertices
                            vertices = []
                            for _ in range(num_vertices):
                                angle = random.uniform(0, 2 * np.pi)
                                radius = random.uniform(poly_size * 0.3, poly_size * 0.5)
                                x = int(center_x + radius * np.cos(angle))
                                y = int(center_y + radius * np.sin(angle))
                                # Clamp to image bounds
                                x = max(0, min(x, img_array.shape[1] - 1))
                                y = max(0, min(y, img_array.shape[0] - 1))
                                vertices.append((x, y))
                            
                            # Draw polygon on image (random color)
                            color = (
                                random.randint(0, 255),
                                random.randint(0, 255),
                                random.randint(0, 255)
                            )
                            img_pil = Image.fromarray(img_array)
                            draw = ImageDraw.Draw(img_pil)
                            draw.polygon(vertices, fill=color, outline=color)
                            img_array = np.array(img_pil)
                            
                            # Update mask: set polygon region to 0 (anomaly)
                            # Only update if the polygon is within road region
                            poly_mask = np.zeros_like(output_mask, dtype=np.uint8)
                            cv2.fillPoly(poly_mask, [np.array(vertices)], 255)
                            poly_mask = (poly_mask > 0)
                            # Only mark as anomaly if it overlaps with road
                            road_overlap = poly_mask & road_mask
                            if road_overlap.any():
                                output_mask[road_overlap] = 0  # Mark as anomaly
        
        # Reset random seed if it was set
        if randseed is not None:
            random.seed()
            np.random.seed()
        
        # Convert back to PIL Images
        augmented_image = Image.fromarray(img_array)
        augmented_mask = Image.fromarray(output_mask, mode='L')
        
        return {
            "image": augmented_image,
            "label": augmented_mask,
        }


