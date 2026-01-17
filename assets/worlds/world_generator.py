import argparse
import random
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import math

def check_collision(new_obs, existing_obstacles):
    """
    Check if new_obs overlaps with any in existing_obstacles.
    obs format: {'type': str, 'pos': [x, y, z], 'size': [d1, d2, ...]}
    """
    x1, y1 = new_obs['pos'][0], new_obs['pos'][1]
    
    for obs in existing_obstacles:
        x2, y2 = obs['pos'][0], obs['pos'][1]
        
        # Calculate distance between centers
        dist = math.hypot(x1 - x2, y1 - y2)
        
        # Determine collision based on types
        if new_obs['type'] == 'cylinder' and obs['type'] == 'cylinder':
            # Cylinder-Cylinder: dist < r1 + r2
            r1 = new_obs['size'][0]
            r2 = obs['size'][0]
            if dist < (r1 + r2):
                return True
                
        elif new_obs['type'] == 'box' and obs['type'] == 'box':
            # Box-Box: AABB overlap
            # Box size is half-extent
            w1, h1 = new_obs['size'][0], new_obs['size'][1]
            w2, h2 = obs['size'][0], obs['size'][1]
            
            if (abs(x1 - x2) < (w1 + w2)) and (abs(y1 - y2) < (h1 + h2)):
                return True
                
        else:
            # Box-Cylinder (mixed)
            # Identify which is which
            if new_obs['type'] == 'cylinder':
                cyl, box = new_obs, obs
                cx, cy = x1, y1
                bx, by = x2, y2
            else:
                cyl, box = obs, new_obs
                cx, cy = x2, y2
                bx, by = x1, y1
                
            r = cyl['size'][0]
            bw, bh = box['size'][0], box['size'][1]
            
            # Find closest point on box to cylinder center
            # Clamp cylinder center to box extents
            closest_x = max(bx - bw, min(cx, bx + bw))
            closest_y = max(by - bh, min(cy, by + bh))
            
            # Distance from cylinder center to closest point
            dist_to_closest = math.hypot(cx - closest_x, cy - closest_y)
            
            if dist_to_closest < r:
                return True
                
    return False

def generate_random_obstacle_data(obstacle_id):
    """
    Generates random obstacle data.
    """
    obs_type = random.choice(['cylinder', 'box'])
    
    # Random position in [-3, 9]
    x = random.uniform(-3, 9)
    y = random.uniform(-3, 9)
    
    if obs_type == 'cylinder':
        radius = random.uniform(0.1, 0.5)
        height = random.uniform(0.5, 0.8)
        half_height = height / 2.0
        z = half_height
        size = [radius, half_height]
        
    else: # box
        half_x = random.uniform(0.1, 0.5)
        half_y = random.uniform(0.1, 0.5)
        height = random.uniform(0.5, 0.8)
        half_z = height / 2.0
        z = half_z
        size = [half_x, half_y, half_z]
        
    return {
        'id': obstacle_id,
        'type': obs_type,
        'pos': [x, y, z],
        'size': size
    }

def format_obstacle_xml(obs):
    x, y, z = obs['pos']
    if obs['type'] == 'cylinder':
        radius, half_height = obs['size']
        xml_str = f"""
    <body name="obstacle_{obs['id']}" pos="{x:.3f} {y:.3f} {z:.3f}">
      <geom type="cylinder" size="{radius:.3f} {half_height:.3f}" rgba="0 0.6 0.2 1" contype="1" conaffinity="1"/>
    </body>
"""
    else:
        half_x, half_y, half_z = obs['size']
        xml_str = f"""
    <body name="obstacle_{obs['id']}" pos="{x:.3f} {y:.3f} {z:.3f}">
      <geom type="box" size="{half_x:.3f} {half_y:.3f} {half_z:.3f}" rgba="0.6 0.2 0 1" contype="1" conaffinity="1"/>
    </body>
"""
    return xml_str

def main():
    parser = argparse.ArgumentParser(description="Generate MuJoCo worlds with random obstacles.")
    parser.add_argument("--num-worlds", type=int, default=5, help="Number of worlds to generate")
    parser.add_argument("--num-obstacles", type=int, default=10, help="Number of obstacles per world")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--template", type=str, default="empty.xml", help="Path to template XML file")
    parser.add_argument("--output-dir", type=str, default="generated_worlds", help="Directory to save generated worlds")

    args = parser.parse_args()

    # Get absolute path for template if it's relative
    script_dir = Path(__file__).parent.resolve()
    if not os.path.isabs(args.template):
        template_path = script_dir / args.template
    else:
        template_path = Path(args.template)

    if not template_path.exists():
        print(f"Error: Template file not found at {template_path}")
        return

    # Create output directory
    if not os.path.isabs(args.output_dir):
        output_dir = script_dir / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read template content
    with open(template_path, 'r') as f:
        template_content = f.read()

    # Adjust relative paths in template if output_dir is different
    # We find all file="..." and meshdir="..." and adjust them
    import re
    
    def replace_path(match):
        attr_name = match.group(1)
        original_path = match.group(2)
        
        # If absolute, don't change
        if os.path.isabs(original_path):
            return match.group(0)
            
        # Resolve absolute path of the target file relative to template dir
        abs_target = (template_path.parent / original_path).resolve()
        
        # Calculate new relative path from output_dir
        try:
            new_path = os.path.relpath(abs_target, output_dir)
        except ValueError:
            # On some systems relpath fails across drives, unlikely here
            return match.group(0)
            
        return f'{attr_name}="{new_path}"'

    template_content = re.sub(r'(file|meshdir)="([^"]+)"', replace_path, template_content)

    # The template has </worldbody> and we want to insert before it.
    # We will look for <worldbody> and </worldbody> to ensure we are inside.
    # A simple string replacement or split might be robust enough if the file structure is known.
    
    if "</worldbody>" not in template_content:
        print("Error: Could not find </worldbody> tag in template.")
        return

    split_content = template_content.split("</worldbody>")
    header = split_content[0]
    footer = "</worldbody>" + split_content[1]

    # Generate worlds
    for i in range(args.num_worlds):
        current_seed = args.seed + i
        random.seed(current_seed)
        
        obstacles_xml = f"\n    <!-- Generated Obstacles (Seed: {current_seed}) -->\n"
        
        existing_obstacles = []
        max_retries = 100
        
        for j in range(args.num_obstacles):
            # Try to populate an obstacle without collision
            valid = False
            for _ in range(max_retries):
                obs_data = generate_random_obstacle_data(j)
                if not check_collision(obs_data, existing_obstacles):
                    existing_obstacles.append(obs_data)
                    valid = True
                    break
            
            if valid:
                obstacles_xml += format_obstacle_xml(obs_data)
            else:
                print(f"Warning: Could not place obstacle {j} in world {current_seed} after {max_retries} retries.")
        
        final_xml = header + obstacles_xml + footer
        
        # Determine difficulty
        if args.num_obstacles < 10:
            difficulty = "easy"
        elif 10 <= args.num_obstacles <= 20:
            difficulty = "medium"
        else:
            difficulty = "hard"

        output_filename = output_dir / f"world_{current_seed}_{difficulty}.xml"
        
        with open(output_filename, 'w') as f:
            f.write(final_xml)
        
        print(f"Generated {output_filename}")

if __name__ == "__main__":
    main()

# python3 world_generator.py --num-worlds 3 --num-obstacles 16 --seed 7 --output-dir train/