from PIL import Image
import glob

frame_files = sorted(glob.glob("lbm_frames/frame_*.png"))
frames = [Image.open(frame) for frame in frame_files]

# Save as an animated GIF.
output_gif = "lbm_simulation.gif"
frames[0].save(
    output_gif,
    save_all=True,
    append_images=frames[1:],
    optimize=True,
    duration=33,
    loop=0
)

print(f"GIF saved as {output_gif}")
