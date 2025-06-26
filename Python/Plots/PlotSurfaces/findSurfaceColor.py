from PIL import Image

# Load the image
im = Image.open("SurfaceHalfPlot_clean.png")
im = im.convert('RGB')

# Get the color of the bottom-left pixel (or any background pixel)
pixel = im.getpixel((0, im.height - 1))  # (x, y)
hex_color = '#{:02x}{:02x}{:02x}'.format(*pixel)

print(f"Bottom-left pixel RGB: {pixel}")
print(f"Hex color: {hex_color}")
