import argparse
from pathlib import Path
import imageio.v2 as iio
import numpy as np
import cv2
import random
import heapq

def parser():
    parser = argparse.ArgumentParser(description='Line Follower')
    parser.add_argument('--input-file', type=Path, help='Path to the masked image file')
    parser.add_argument('--mask-value', type=int, default=255, help='Pixel value to identify the line in the masked image')
    return parser.parse_args()


def get_neighbors(point, delta):
    """Returns coordinates in a square radius, excluding the center."""
    x, y = point
    for dx in range(-delta, delta + 1):
        for dy in range(-delta, delta + 1):
            if dx == 0 and dy == 0: continue
            yield (x + dx, y + dy), np.sqrt(dx**2 + dy**2)

def follow_grain_bidirectional(masked_image, start_point, jump_dist=15):
    """
    Explores a line in all directions from a seed point using a 
    Cost-based priority search (A* style).
    """
    # Use a priority queue: (cost, x, y)
    # Lower cost (distance) is prioritized first
    frontier = []
    heapq.heappush(frontier, (0, start_point[0], start_point[1]))
    
    emulsion_path = []
    visited = set()
    visited.add(start_point)

    while frontier:
        # Pop the point with the lowest 'jump cost'
        current_cost, curr_x, curr_y = heapq.heappop(frontier)
        emulsion_path.append((curr_x, curr_y))

        # Look for neighbors within jump_dist
        for (next_pt, dist) in get_neighbors((curr_x, curr_y), jump_dist):
            nx, ny = next_pt
            
            # Boundary check
            if 0 <= nx < masked_image.shape[1] and 0 <= ny < masked_image.shape[0]:
                if masked_image[ny, nx] == 255 and next_pt not in visited:
                    visited.add(next_pt)
                    # We add the distance to the cost to prefer local 'soldering'
                    heapq.heappush(frontier, (dist, nx, ny))
                    
                    # Optimization: If we found an immediate neighbor (dist < 1.5), 
                    # we stop looking for further jumps for this specific step.
                    if dist < 1.5:
                        break 

    return emulsion_path


def follow_line(masked_image, start_point, film_coordinate, mask_value=255):
    current_point = start_point
    emulsion = [start_point]
    visited = {start_point}  # Memory to prevent infinite loops
    
    delta = 50 # Stick to neighbors to follow the "line" precisely
    
    while True:
        x, y = current_point
        found_next_point = False
        
        # Check surrounding pixels
        for dx in range(-delta, delta + 1):
            for dy in range(-delta, delta + 1):
                if dx == 0 and dy == 0: continue # Don't check yourself
                
                next_x, next_y = x + dx, y + dy
                candidate = (int(next_x), int(next_y))
                
                # Logic: Is it in the film, is it white, and HAVE WE NOT been there?
                if (candidate in film_coordinate and 
                    masked_image[next_y, next_x] == mask_value and 
                    candidate not in visited):
                    
                    current_point = candidate
                    emulsion.append(current_point)
                    visited.add(candidate) # Record the visit
                    found_next_point = True
                    break
            if found_next_point: break
            
        if not found_next_point:
            break
            
    return emulsion


def thin_and_display_mask(mask_path: Path):
    """
    Loads a binary mask, thins it to a 1-pixel skeleton, 
    and displays the result to verify the spiral path.
    """
    # 1. Load the mask in grayscale
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not load mask at {mask_path}")
        return

    # 2. Ensure the mask is strictly binary (0 or 255)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 3. Perform Thinning (Skeletonization)
    # This reduces the ribbon to a single-pixel-wide line
    skeleton = cv2.ximgproc.thinning(binary)

    return skeleton


def extract_full_grain(masked_image, start_point, film_coordinate, mask_value=255):
    stack = [start_point]
    emulsion = []
    visited = {start_point}
    
    delta = 1
    
    while stack:
        # 'Pop' behaves like the return of a recursive call
        current_x, current_y = stack.pop()
        emulsion.append((current_x, current_y))
        
        # Check all 8 directions (the branching factor)
        for dx in range(-delta, delta + 1):
            for dy in range(-delta, delta + 1):
                if dx == 0 and dy == 0: continue
                
                next_x, next_y = current_x + dx, current_y + dy
                candidate = (int(next_x), int(next_y))
                
                # Check boundaries and mask value
                if (0 <= next_x < masked_image.shape[1] and 
                    0 <= next_y < masked_image.shape[0] and
                    candidate in film_coordinate and
                    masked_image[next_y, next_x] == mask_value and 
                    candidate not in visited):
                    
                    visited.add(candidate)
                    stack.append(candidate) # Add to stack to explore later
                    print(f'Adding point to stack: {candidate}')
    print(f'Emulsion points collected: {len(emulsion)}')
                    
    return emulsion

def select_a_random_point(masked_image, mask_value=255, display_canvas=False):
    print(f'Shape of masked image: {masked_image.shape}')
    h, w = masked_image.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:,:,0] = masked_image # BGR
    film_coordinate = np.where(masked_image == mask_value)
    film_coordinate = list(zip(film_coordinate[1], film_coordinate[0]))
    print(film_coordinate)
    start_x, start_y = film_coordinate[random.randint(0, len(film_coordinate) - 1)] # True random point on the line
    print(f'Starting point: ({start_x}, {start_y})')
    a_random_point = (start_x, start_y)
    if display_canvas:
        cv2.circle(canvas, (start_x, start_y), 5, (0, 255, 0), -1)
        cv2.imshow('Canvas', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return a_random_point, canvas, film_coordinate

import heapq
import numpy as np

def extract_contours_from_mask(masked_image, jump_dist=10):
    """
    Finds all separate objects in a mask and extracts their 
    ordered pointsets using a jump-aware pathfinder.
    """
    h, w = masked_image.shape
    visited = np.zeros_like(masked_image, dtype=bool)
    all_contours = []

    # 1. Global Scan for a new 'seed' point
    for y in range(h):
        for x in range(w):
            if masked_image[y, x] == 255 and not visited[y, x]:
                # Found a new object! Start the pathfinder
                contour = trace_single_contour(masked_image, (x, y), visited, jump_dist)
                if len(contour) > 2: # Ignore single noise pixels
                    all_contours.append(contour)
    
    return all_contours
def find_ordered_contour(masked_image, start_point, jump_dist=15):
    """
    Traces a line pixel-by-pixel to ensure a head-to-tail ordered sequence.
    If it hits a gap, it 'solders' to the nearest unvisited pixel.
    """
    h, w = masked_image.shape
    ordered_points = [start_point]
    visited = np.zeros_like(masked_image, dtype=bool)
    visited[start_point[1], start_point[0]] = True
    
    current_pt = start_point
    
    while True:
        curr_x, curr_y = current_pt
        found_next = False
        
        # 1. Search immediate neighbors first (Radius 1)
        # We check these in a specific circular order to maintain 'flow'
        neighbors = [
            (0, 1), (1, 1), (1, 0), (1, -1), 
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        
        for dx, dy in neighbors:
            nx, ny = curr_x + dx, curr_y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if masked_image[ny, nx] == 255 and not visited[ny, nx]:
                    current_pt = (nx, ny)
                    visited[ny, nx] = True
                    ordered_points.append(current_pt)
                    found_next = True
                    break
        
        # 2. If stuck, try to "Solder" (Jump the Gap)
        if not found_next:
            best_jump_pt = None
            min_dist = float('inf')
            
            # Look in expanding squares for the nearest NEW island
            for r in range(2, jump_dist + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        # Only check the perimeter of the square radius r
                        if abs(dx) != r and abs(dy) != r: continue
                        
                        nx, ny = curr_x + dx, curr_y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if masked_image[ny, nx] == 255 and not visited[ny, nx]:
                                d = np.sqrt(dx**2 + dy**2)
                                if d < min_dist:
                                    min_dist = d
                                    best_jump_pt = (nx, ny)
                
                if best_jump_pt: # Found the closest pixel on the next island
                    break
            
            if best_jump_pt:
                current_pt = best_jump_pt
                visited[best_jump_pt[1], best_jump_pt[0]] = True
                ordered_points.append(current_pt)
                found_next = True
        
        # 3. If still nothing found after jumping, we are done with this contour
        if not found_next:
            break
            
    return ordered_points

def trace_single_contour(masked_image, start_node, visited_map, jump_dist):
    """Uses Priority Queue to solder gaps and trace a single object."""
    h, w = masked_image.shape
    contour_points = []
    pq = [(0, start_node[0], start_node[1])] # (cost, x, y)
    visited_map[start_node[1], start_node[0]] = True

    while pq:
        cost, curr_x, curr_y = heapq.heappop(pq)
        contour_points.append((curr_x, curr_y))

        # Search in a window defined by jump_dist
        for dy in range(-jump_dist, jump_dist + 1):
            for dx in range(-jump_dist, jump_dist + 1):
                nx, ny = curr_x + dx, curr_y + dy
                
                if 0 <= nx < w and 0 <= ny < h:
                    if masked_image[ny, nx] == 255 and not visited_map[ny, nx]:
                        dist = np.sqrt(dx**2 + dy**2)
                        visited_map[ny, nx] = True
                        heapq.heappush(pq, (dist, nx, ny))
                        
                        # Optimization: if it's an immediate neighbor, 
                        # stop searching for further jumps for this pixel
                        if dist < 1.5: 
                            break
    return contour_points

def main():
    args = parser()
    input_file = args.input_file
    mask_value = args.mask_value

    if not input_file.is_file():
        print(f"Error: {input_file} does not exist.")
        return

    # Here you would add the code to process the masked image and follow the line.
    # This is a placeholder for the actual line following logic.
    print(f"Processing masked image from: {input_file}")
    masked_image = thin_and_display_mask(input_file)
    canvas = np.zeros((masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8)
    canvas[:,:,0] = masked_image # BGR
    all_contours = find_ordered_contour(masked_image, select_a_random_point(masked_image, mask_value)[0], jump_dist=15)
    print(f"Extracted {len(all_contours[0])} contours from the mask.")
    print("contours1:", all_contours[0])
    for i, contour in enumerate(all_contours):
        # print(f'Contour point {i}: {contour}, G: {int((1-(i/len(all_contours[0])))*255)}, R: {int(i/len(all_contours[0])*255)}, B: 0')
        cv2.circle(canvas, contour, 2, (0, int((1-(i/len(all_contours[0])))*255), int(i/len(all_contours[0])*255)), -1)
        # break
    # contours2 = all_contours[1]
    # print("contours2:", contours2)
    # for i, contour in enumerate(all_contours):
    #     # print(f'Contour point {i}: {contour}, G: {int((1-(i/len(all_contours[0])))*255)}, R: {int(i/len(all_contours[0])*255)}, B: 0')
    #     cv2.circle(canvas, contour, 2, (0, int((1-(i/len(all_contours[0])))*255), int(i/len(all_contours[0])*255)), -1)
    #     # break
    cv2.imwrite('/Volumes/Ankan_PhD/segmentation mask/output_contours.png', canvas)



if __name__ == "__main__":
    main()