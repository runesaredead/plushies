from PIL import Image
import json
import os
import numpy as np
from scipy.ndimage import gaussian_filter

def load_spritesheet(path):
    return Image.open(path)

def split_spritesheet(sheet, rows=6, cols=6, top_crop=5):
    width = sheet.width // cols
    height = sheet.height // rows
    sprites = []
    
    for row in range(rows):
        for col in range(cols):
            left = col * width
            top = row * height
            right = left + width
            bottom = top + height
            
            # Crop the sprite
            sprite = sheet.crop((left, top, right, bottom))
            
            # Apply additional top crop if needed
            if top_crop > 0:
                sprite_width, sprite_height = sprite.size
                if sprite_height > top_crop * 2:  # Make sure we don't crop too much
                    sprite = sprite.crop((0, top_crop, sprite_width, sprite_height))
            
            sprites.append(sprite)
    
    return sprites

def calculate_thickness_map(sprite_array, smooth_factor=1.5):
    """Calculate the thickness of the plushie with enhanced smoothing"""
    # Create alpha mask
    alpha_mask = sprite_array[:, :, 3] > 128
    
    # Smooth the edges more for simpler appearance
    smoothed = gaussian_filter(alpha_mask.astype(float), sigma=smooth_factor)
    
    # Calculate distance from edges for puffiness
    height, width = smoothed.shape
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.max(dist_from_center)
    
    # Create base thickness with higher minimum thickness
    thickness = smoothed.copy()
    
    # Add extra thickness to solid areas
    thickness[thickness > 0.5] += 0.4
    
    # Normalize
    thickness = thickness / np.max(thickness)
    
    # Increase minimum thickness
    thickness = thickness * 0.7 + 0.3
    
    return thickness

def add_puffy_voxels(voxels, x, y, z, r, g, b, darkness, width, height, local_thickness):
    """Add puffy voxels around a point in all relevant directions"""
    base_color = [
        int(r * darkness * 0.95),
        int(g * darkness * 0.95),
        int(b * darkness * 0.95)
    ]
    
    # Side puffs (X and Y directions)
    for dx, dy in [(-0.5,0), (0.5,0), (0,-0.5), (0,0.5)]:
        voxels.append({
            'position': [
                (x + dx) * 2,
                (height - 1 - (y + dy)) * 2,
                z * 2
            ],
            'color': base_color.copy()
        })
    
    # Z-direction puffs (front and back)
    if z != 0:  # Only add Z puffs if not at center
        z_offset = 0.5 if z > 0 else -0.5
        voxels.append({
            'position': [
                x * 2,
                (height - 1 - y) * 2,
                (z + z_offset) * 2
            ],
            'color': base_color.copy()
        })

def create_3d_voxels(sprite, resolution=80, depth=9):
    """Create 3D voxels by mirroring the front half to the back half with improved rounding"""
    # Resize sprite to target resolution
    sprite = sprite.resize((resolution, resolution), Image.Resampling.LANCZOS)
    sprite_array = np.array(sprite)
    
    # Check if sprite has any opaque pixels
    if np.max(sprite_array[:, :, 3]) <= 128:
        print("  Warning: Sprite appears to be fully transparent")
    
    width, height = sprite.size
    voxels = []
    
    # Calculate thickness map with stronger smoothing for rounder surfaces
    thickness = calculate_thickness_map(sprite_array, smooth_factor=1.8)
    
    # Adjust depth based on sprite size (for larger characters)
    actual_pixel_count = np.sum(sprite_array[:, :, 3] > 128)
    fill_ratio = actual_pixel_count / (width * height)
    
    # Scale depth slightly based on fill ratio - fuller sprites get slightly less depth
    adjusted_depth = max(5, min(depth, round(depth * (1.1 - fill_ratio * 0.2))))
    if adjusted_depth != depth:
        print(f"  Adjusting depth to {adjusted_depth} (fill ratio: {fill_ratio:.2f})")
    
    # FIRST PASS: Create only the front half voxels
    front_voxels = []
    for y in range(height):
        for x in range(width):
            r, g, b, a = sprite_array[y, x]
            if a > 128:
                # Get thickness at this point
                local_thickness = thickness[y, x]
                
                # Calculate local depth (only need half the depth now)
                local_depth = int(adjusted_depth * local_thickness * 0.5)  # Reduced since we'll mirror
                if local_depth < 2:  # Ensure minimum thickness for half
                    local_depth = 2
                    
                # Create only the front half (z >= 0)
                for z in range(0, local_depth + 1):
                    # Calculate darkness based on distance from center
                    z_center_dist = z / local_depth  # Only front half, so different calculation
                    darkness = 1.0 - (z_center_dist * 0.15)
                    
                    # Slightly simplify colors for more consistent surface
                    simplified_r = int(r * darkness / 10) * 10
                    simplified_g = int(g * darkness / 10) * 10
                    simplified_b = int(b * darkness / 10) * 10
                    
                    # Colors for this layer
                    color = [
                        simplified_r,
                        simplified_g,
                        simplified_b
                    ]
                    
                    # Add front voxel
                    voxel = {
                        'position': [x * 2, (height - 1 - y) * 2, z * 2],
                        'color': color.copy()
                    }
                    front_voxels.append(voxel)
    
    # Second pass: Add surface detail voxels just for the front half
    for y in range(height):
        for x in range(width):
            r, g, b, a = sprite_array[y, x]
            if a > 128:
                # Only add detail to edge pixels
                is_edge = False
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        is_edge = True
                        break
                    if sprite_array[ny, nx, 3] <= 128:
                        is_edge = True
                        break
                
                if is_edge:
                    local_thickness = thickness[y, x]
                    local_depth = int(adjusted_depth * local_thickness * 0.5)
                    if local_depth < 2:
                        local_depth = 2
                    
                    # Add middle surface details just for the front
                    add_puffy_voxels_simplified(
                        front_voxels, x, y, 0,
                        r, g, b, 0.9,
                        width, height,
                        local_thickness
                    )
    
    # Add all front voxels to the main voxel list
    voxels.extend(front_voxels)
    
    # THIRD PASS: Mirror the front voxels to create back half
    for voxel in front_voxels:
        # Get the original position
        x, y, z = voxel['position']
        
        # Skip the center plane (z=0) to avoid duplicates
        if z == 0:
            continue
            
        # Create mirrored position (negative z)
        mirrored_position = [x, y, -z]
        
        # Use the same color as the front
        mirrored_voxel = {
            'position': mirrored_position,
            'color': voxel['color'].copy()
        }
        
        voxels.append(mirrored_voxel)
    
    return voxels

def add_puffy_voxels_simplified(voxels, x, y, z, r, g, b, darkness, width, height, local_thickness):
    """Add simplified puffy voxels only at key points for visual detail"""
    # Simplify colors
    simplified_r = int(r * darkness / 10) * 10
    simplified_g = int(g * darkness / 10) * 10
    simplified_b = int(b * darkness / 10) * 10
    
    base_color = [
        simplified_r,
        simplified_g,
        simplified_b
    ]
    
    # Add single central voxel for smoother appearance without ridges
    voxels.append({
        'position': [
            x * 2,
            (height - 1 - y) * 2,
            z * 2
        ],
        'color': base_color.copy()
    })

def add_puffy_voxels_tapered(voxels, x, y, z, r, g, b, darkness, width, height, local_thickness, is_front):
    """Add tapered puffy voxels for the front and back layers"""
    # Simplify colors
    simplified_r = int(r * darkness / 10) * 10
    simplified_g = int(g * darkness / 10) * 10
    simplified_b = int(b * darkness / 10) * 10
    
    base_color = [
        simplified_r,
        simplified_g,
        simplified_b
    ]
    
    # Only add back voxels - no longer adding front voxels
    if not is_front:
        z_offset = -0.5  # Only used for back voxels now
        voxels.append({
            'position': [
                x * 2,
                (height - 1 - y) * 2,
                (z + z_offset) * 2
            ],
            'color': base_color.copy()
        })

def save_voxel_file(voxels, filename):
    """Save voxels in an optimized format:
    - positions: flat array of [x,y,z] coordinates
    - colors: palette of unique colors
    - color_indices: indices into the color palette for each voxel
    """
    # Extract positions and colors
    positions = []
    colors = []
    color_map = {}  # Map from color tuple to index
    color_indices = []
    
    for voxel in voxels:
        # Add position as flat array
        pos = voxel['position']
        positions.extend([int(pos[0]), int(pos[1]), int(pos[2])])
        
        # Add color to palette if new
        color = tuple(voxel['color'])
        if color not in color_map:
            color_map[color] = len(color_map)
            colors.extend(color)
        
        # Store color index
        color_indices.append(color_map[color])
    
    # Create optimized data structure
    data = {
        'p': positions,  # Flat array of coordinates [x1,y1,z1,x2,y2,z2,...]
        'c': colors,     # Flat array of unique colors [r1,g1,b1,r2,g2,b2,...]
        'i': color_indices,  # Array of indices into color palette
        'm': {
            'voxel_count': len(voxels),
            'unique_colors': len(color_map)
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    
    actual_size = os.path.getsize(filename)
    print(f'  Voxels: {len(voxels)}, Unique colors: {len(color_map)}')
    print(f'  File size: {actual_size/1024:.1f}KB')

def process_spritesheet(sheet_path, sheet_number=0, base_index=0, rows=6, cols=6, top_crop=5):
    """Process a single spritesheet and generate voxel models"""
    # Load and split spritesheet
    try:
        sheet = load_spritesheet(sheet_path)
        if sheet is None:
            print(f"Failed to load sheet: {sheet_path}")
            return 0
            
        # Validate sheet dimensions
        expected_width = cols * sheet.width // cols  # Should equal sheet.width if cols divides evenly
        expected_height = rows * sheet.height // rows  # Should equal sheet.height if rows divides evenly
        
        if sheet.width != expected_width or sheet.height != expected_height:
            print(f"WARNING: Sheet {sheet_path} dimensions ({sheet.width}x{sheet.height}) may not be exactly divisible by grid ({rows}x{cols})")
            
        sprites = split_spritesheet(sheet, rows, cols, top_crop)
        total_sprites = len(sprites)
        
        print(f'Processing {total_sprites} sprites from {sheet_path}...\n')
        
        success_count = 0
        for i, sprite in enumerate(sprites):
            sprite_index = base_index + i
            try:
                print(f'[{i+1}/{total_sprites}] Processing sprite {sprite_index}...')
                voxels = create_3d_voxels(sprite, resolution=80, depth=9)
                filename = f'voxels/sprite_{sprite_index:03d}.vox.json'
                save_voxel_file(voxels, filename)
                success_count += 1
                
            except Exception as e:
                print(f'Error processing sprite {sprite_index}: {str(e)}')
                
        print(f"Successfully processed {success_count}/{total_sprites} sprites from {sheet_path}")
        return total_sprites
        
    except Exception as e:
        print(f'Error in spritesheet processing for {sheet_path}: {str(e)}')
        return 0

def main():
    # Create output directory
    os.makedirs('voxels', exist_ok=True)
    
    # Process all sheets
    total_sprites = 0
    total_size = 0
    total_voxels = 0
    
    # Define sheets to process with proper grid dimensions
    sheets = [
        {'path': 'pizzasheet.png', 'sheet_number': 0, 'rows': 6, 'cols': 6},
        {'path': 'pizzasheet2.png', 'sheet_number': 1, 'rows': 7, 'cols': 7},
        {'path': 'pizzasheet3.png', 'sheet_number': 2, 'rows': 7, 'cols': 7},
        {'path': 'pizzasheet4.png', 'sheet_number': 3, 'rows': 7, 'cols': 7},
        {'path': 'pizzasheet5.png', 'sheet_number': 4, 'rows': 7, 'cols': 7}
    ]
    
    # Calculate base indices correctly
    base_index = 0
    for i, sheet_info in enumerate(sheets):
        sheet_info['base_index'] = base_index
        if i < len(sheets) - 1:  # Don't need to calculate for the last sheet
            base_index += sheet_info['rows'] * sheet_info['cols']
    
    for sheet_info in sheets:
        path = sheet_info['path']
        sheet_number = sheet_info['sheet_number']
        base_index = sheet_info['base_index']
        rows = sheet_info['rows']
        cols = sheet_info['cols']
        
        try:
            print(f'Processing sheet {path} with grid {rows}x{cols}...')
            
            # Check if file exists before processing
            if not os.path.exists(path):
                print(f"WARNING: Sheet file {path} does not exist. Skipping.")
                continue
                
            sprites_in_sheet = process_spritesheet(path, sheet_number, base_index, rows, cols, top_crop=5)
            total_sprites += sprites_in_sheet
        except Exception as e:
            print(f'Error processing sheet {path}: {str(e)}')
    
    # Calculate total size and voxel count
    for filename in os.listdir('voxels'):
        if filename.endswith('.vox.json'):
            file_path = os.path.join('voxels', filename)
            total_size += os.path.getsize(file_path)
            
            # Load the file to count voxels
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    total_voxels += data['m']['voxel_count']
            except:
                pass
    
    # Print summary
    print('\nProcessing complete!')
    print(f'Total sheets processed: {len(sheets)}')
    print(f'Total sprites processed: {total_sprites}')
    print(f'Total voxels created: {total_voxels:,}')
    print(f'Total file size: {total_size/1024/1024:.2f}MB')
    print(f'Average file size: {(total_size/total_sprites)/1024:.1f}KB per sprite')

if __name__ == '__main__':
    main() 