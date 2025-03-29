from PIL import Image
import json
import os
import numpy as np
from scipy.ndimage import gaussian_filter

def load_spritesheet(path):
    return Image.open(path)

def split_spritesheet(sheet, rows=6, cols=6):
    width = sheet.width // cols
    height = sheet.height // rows
    sprites = []
    
    for row in range(rows):
        for col in range(cols):
            left = col * width
            top = row * height
            right = left + width
            bottom = top + height
            sprite = sheet.crop((left, top, right, bottom))
            sprites.append(sprite)
    
    return sprites

def calculate_thickness_map(sprite_array):
    """Calculate the thickness of the plushie at each point"""
    # Create alpha mask
    alpha_mask = sprite_array[:, :, 3] > 128
    
    # Smooth the edges
    smoothed = gaussian_filter(alpha_mask.astype(float), sigma=1.5)
    
    # Calculate distance from edges for puffiness
    height, width = smoothed.shape
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.max(dist_from_center)
    
    # Create base thickness
    thickness = smoothed.copy()
    
    # Add extra thickness to solid areas
    thickness[thickness > 0.5] += 0.3
    
    # Normalize
    thickness = thickness / np.max(thickness)
    
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

def create_3d_voxels(sprite, resolution=80, depth=7):
    """Create 3D voxels with perfect symmetry by mirroring the front half"""
    # Resize sprite to target resolution
    sprite = sprite.resize((resolution, resolution), Image.Resampling.LANCZOS)
    sprite_array = np.array(sprite)
    
    width, height = sprite.size
    voxels = []
    
    # Calculate thickness map
    thickness = calculate_thickness_map(sprite_array)
    
    # Create voxels for front half only, then mirror
    for y in range(height):
        for x in range(width):
            r, g, b, a = sprite_array[y, x]
            if a > 128:
                # Get thickness at this point
                local_thickness = thickness[y, x]
                
                # Calculate local depth (thicker in solid areas)
                local_depth = int(depth * local_thickness)
                # Only generate front half (z >= 0)
                start_z = 0
                end_z = local_depth // 2 + 1
                
                # Create front voxels and their mirrors
                for z in range(start_z, end_z):
                    # Calculate darkness based on distance from center
                    z_center_dist = abs(z) / (local_depth / 2)
                    darkness = 1.0 - (z_center_dist * 0.2)
                    
                    # Colors for this layer
                    color = [
                        int(r * darkness),
                        int(g * darkness),
                        int(b * darkness)
                    ]
                    
                    # Add front voxel
                    front_voxel = {
                        'position': [x * 2, (height - 1 - y) * 2, z * 2],
                        'color': color.copy()
                    }
                    voxels.append(front_voxel)
                    
                    # Add mirrored back voxel (except at z=0 to avoid duplicates)
                    if z > 0:
                        back_voxel = {
                            'position': [x * 2, (height - 1 - y) * 2, -z * 2],
                            'color': color.copy()
                        }
                        voxels.append(back_voxel)
                    
                    # Add puffy voxels in all directions if near an edge
                    is_edge = (
                        x == 0 or x == width-1 or  # Left/right edges
                        y == 0 or y == height-1 or  # Top/bottom edges
                        z == end_z-1 or  # Front edge
                        local_thickness < 0.8  # Near any edge in thickness map
                    )
                    
                    if is_edge:
                        # Add front puffy voxels
                        add_puffy_voxels_half(
                            voxels, x, y, z,
                            r, g, b, darkness,
                            width, height,
                            local_thickness,
                            True  # front half
                        )
                        
                        # Add mirrored back puffy voxels (except at z=0)
                        if z > 0:
                            add_puffy_voxels_half(
                                voxels, x, y, -z,
                                r, g, b, darkness,
                                width, height,
                                local_thickness,
                                False  # back half
                            )
    
    return voxels

def add_puffy_voxels_half(voxels, x, y, z, r, g, b, darkness, width, height, local_thickness, is_front):
    """Add puffy voxels for half the model"""
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
    
    # Z-direction puffs (only in the direction we're building)
    if is_front and z < 7:  # Front half
        voxels.append({
            'position': [
                x * 2,
                (height - 1 - y) * 2,
                (z + 0.5) * 2
            ],
            'color': base_color.copy()
        })
    elif not is_front and z > -7:  # Back half
        voxels.append({
            'position': [
                x * 2,
                (height - 1 - y) * 2,
                (z - 0.5) * 2
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

def process_spritesheet(sheet_path, sheet_number=0, base_index=0, rows=6, cols=6):
    """Process a single spritesheet and generate voxel models"""
    # Load and split spritesheet
    sheet = load_spritesheet(sheet_path)
    sprites = split_spritesheet(sheet, rows, cols)
    total_sprites = len(sprites)
    
    print(f'Processing {total_sprites} sprites from {sheet_path}...\n')
    
    for i, sprite in enumerate(sprites):
        sprite_index = base_index + i
        try:
            print(f'[{i+1}/{total_sprites}] Processing sprite {sprite_index}...')
            voxels = create_3d_voxels(sprite, resolution=80, depth=7)
            filename = f'voxels/sprite_{sprite_index:03d}.vox.json'
            save_voxel_file(voxels, filename)
            
        except Exception as e:
            print(f'Error processing sprite {sprite_index}: {str(e)}')
    
    return total_sprites

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
        {'path': 'pizzasheet4.png', 'sheet_number': 3, 'rows': 7, 'cols': 7}
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
            sprites_in_sheet = process_spritesheet(path, sheet_number, base_index, rows, cols)
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