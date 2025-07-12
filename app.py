base_path="."
import subprocess
import torch
import gc
import BEN2  
from PIL import Image
import os 
import re 
import uuid
import uuid
import shutil
from zipfile import ZipFile



def create_folder():
  global base_path
  os.makedirs(f"{base_path}/result/",exist_ok=True)
  os.makedirs(f"{base_path}/temp/",exist_ok=True)


def get_max_gpu_memory():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        return round(total_memory / (1024 ** 3), 2)  # Convert to GB
    return None

def is_gpu_memory_over_limit():
    max_gpu_memory = get_max_gpu_memory()
    if max_gpu_memory is None:
        print("CUDA is not available.")
        return False
    
    limit_gb = max_gpu_memory - 0.60  # Set limit just below max GPU memory
    
    # Run `nvidia-smi` to get memory usage
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                            stdout=subprocess.PIPE, text=True)
    
    memory_used_mb_list = result.stdout.strip().splitlines()
    
    for i, memory_used_mb in enumerate(memory_used_mb_list):
        memory_used_gb = int(memory_used_mb) / 1024.0
        if memory_used_gb > limit_gb:
            return True  # GPU memory exceeded
    
    return False


# Load the model
def load_model():
    global model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    model = BEN2.BEN_Base().to(device).eval() #init pipeline
    model.loadcheckpoints("./BEN2_Base.pth")
    return model


def clean_file_name(file_path):
    # Get the base file name and extension
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)

    # Replace non-alphanumeric characters with an underscore
    cleaned = re.sub(r'[^a-zA-Z\d]+', '_', file_name)

    # Remove any multiple underscores
    clean_file_name = re.sub(r'_+', '_', cleaned).strip('_')

    # Generate a random UUID for uniqueness
    random_uuid = uuid.uuid4().hex[:6]

    # Combine cleaned file name with the original extension
    clean_file_name=clean_file_name + f"_{random_uuid}" 
    return clean_file_name,file_extension





def remove_background_from_image(image_path,high_quality_matting=True):
  global base_path,model
  # Check GPU memory and reload if necessary
  if is_gpu_memory_over_limit():
      model = load_model()
      print("Model reloaded due to high GPU memory usage.")

  image = Image.open(image_path)
  foreground = model.inference(image, refine_foreground=high_quality_matting) #Refine foreground is an extract postprocessing step that increases inference time but can improve matting edges. The default value is False.
  save_path=f"{base_path}/result/{clean_file_name(image_path)[0]}.png"
  foreground.save(save_path)
  return save_path


def zip_folder(folder_path, zip_path):
    if os.path.exists(zip_path):
      os.remove(zip_path)
    with ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname=arcname)

def manage_files(multiple_images):
    if not multiple_images:
        return None, None

    if len(multiple_images) == 1:
        save_path = remove_background_from_image(multiple_images[0])
        return save_path, Image.open(save_path)

    # For multiple images, process and zip results
    random_uuid = uuid.uuid4().hex[:6]
    temp_folder=f"{base_path}/temp/BEN2_{random_uuid}"
    os.makedirs(temp_folder,exist_ok=True)

    for image in multiple_images:
        try:
            save_path = remove_background_from_image(image)
            shutil.move(save_path, temp_folder)
        except Exception as e:
            print(f"Skipping {image}: {e}")

    zip_folder(temp_folder, f"{temp_folder}.zip")
    zip_path = os.path.abspath(f"{temp_folder}.zip")

    shutil.rmtree(temp_folder)  # Clean up temp folder after zipping

    return zip_path, None


def remove_background_from_video(video_path,high_quality_matting=True,batch_size=1,print_frame_process=False,
                                 background_color=(0, 255, 0)):
  global base_path,model
  model = load_model()
  save_folder=f"{base_path}/result"
  save_path=f"{save_folder}/{clean_file_name(video_path)[0]}.mp4"
  try:
    model.segment_video(
      video_path= video_path,
      output_path=save_folder, # Outputs will be saved as foreground.webm or foreground.mp4. The default value is "./"
      fps=0, # If this is set to 0 CV2 will detect the fps in the original video. The default value is 0.
      refine_foreground=high_quality_matting,  #refine foreground is an extract postprocessing step that increases inference time but can improve matting edges. The default value is False.
      batch=batch_size,  # We recommended that batch size not exceed 3 for consumer GPUs as there are minimal inference gains. The default value is 1.
      print_frames_processed=print_frame_process,  #Informs you what frame is being processed. The default value is True.
      webm = False, # This will output an alpha layer video but this defaults to mp4 when webm is false. The default value is False.
      rgb_value= background_color # If you do not use webm this will be the RGB value of the resulting background only when webm is False. The default value is a green background (0,255,0).
    )
    generated_video=f"{save_folder}/foreground_output_with_audio.mp4"
    if os.path.exists(generated_video):
      shutil.move(generated_video,save_path)
      return save_path,save_path
  except Exception as e:
    print(e)
    return None,None



create_folder()  
# Define the device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize model
model = BEN2.BEN_Base().to(device).eval() #init pipeline
model.loadcheckpoints("./BEN2_Base.pth")
#@title Run Gradio Interface
#@title Run Gradio Interface
import gradio as gr
import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
  description = """BEN2 for Background Removal<br>
  <span style='font-size: 16px;'>HuggingFace model page: <a href='https://huggingface.co/PramaLLC/BEN2' target='_blank'>BEN2</a></span>
  """
  # Define Gradio inputs and outputs
  image_demo_inputs=[gr.File(label="Upload Single or Multiple Images",file_count="multiple",file_types=['image'],type='filepath')]
  image_demo_outputs=[gr.File(label="Download Image or Zip File", show_label=True),gr.Image(label="Result")]
  image_demo = gr.Interface(fn=manage_files, inputs=image_demo_inputs,outputs=image_demo_outputs,title="Remove Image Background (Support Batch image processing)")
  video_demo_inputs=[gr.File(label="Upload a Video",file_types=['.mp4'],type='filepath')]
  video_demo_outputs=[gr.File(label="Download Video", show_label=True),gr.Video(label="Green Screen Video")]
  video_demo = gr.Interface(fn=remove_background_from_video, inputs=video_demo_inputs,outputs=video_demo_outputs, title="Remove Video Background (Make Green Screen Video)")
  demo = gr.TabbedInterface([image_demo,video_demo], ["Remove Image Background", "Remove Video Background"],title=description)
  demo.queue().launch(allowed_paths=[f"{base_path}/result",f"{base_path}/temp",base_path],debug=debug,share=share)
if __name__ == "__main__":
    main()


