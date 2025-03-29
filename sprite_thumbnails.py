from PIL import Image
import os

def extract_thumbnails_from_sheet(sheet_path, sheet_number=0, base_index=0, rows=6, cols=6, top_crop=5):
    """Extract thumbnails from a single spritesheet"""
    # Create thumbnails directory if it doesn't exist
    os.makedirs('thumbnails', exist_ok=True)
    
    # Load the spritesheet
    try:
        sheet = Image.open(sheet_path)
    except Exception as e:
        print(f"Error loading sheet {sheet_path}: {str(e)}")
        return 0
    
    # Define grid parameters based on input parameters
    width = sheet.width // cols
    height = sheet.height // rows
    
    # Extract each sprite and save as a thumbnail
    count = 0
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
            
            # Save as thumbnail with global index
            sprite_index = base_index + count
            filename = f'thumbnails/sprite_{sprite_index:03d}.png'
            sprite.save(filename)
            print(f"Saved {filename}")
            
            count += 1
    
    return count

def extract_thumbnails():
    """Extract thumbnails from all pizzasheet files"""
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
    
    total_thumbnails = 0
    
    for sheet_info in sheets:
        path = sheet_info['path']
        sheet_number = sheet_info['sheet_number']
        base_index = sheet_info['base_index']
        rows = sheet_info['rows']
        cols = sheet_info['cols']
        
        print(f"\nProcessing sheet {path} with grid {rows}x{cols}...")
        thumbnails_in_sheet = extract_thumbnails_from_sheet(path, sheet_number, base_index, rows, cols, top_crop=5)
        total_thumbnails += thumbnails_in_sheet
    
    print(f"\nTotal thumbnails extracted: {total_thumbnails}")

if __name__ == "__main__":
    extract_thumbnails() 