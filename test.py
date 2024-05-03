import sys
import subprocess
from rvc_python.infer import infer_file, infer_files
from uuid import uuid4
# Mock the arguments

models_dict = {"Lexa": 'models/lexa.pth'}
# Assuming the above script is in a module named 'voice_conversion_module'
def infer(input, output, model):
    infer_file(
    input_path=input,
    model_path=model,
    index_path="",  # Optional: specify path to index file if available
    device="cuda:0", # Use cpu or cuda
    f0method="harvest",  # Choose between 'harvest', 'crepe', 'rmvpe', 'pm'
    opt_path=output,  # Output file path
    index_rate=0.5,
    filter_radius=3,
    resample_sr=0,  # Set to desired sample rate or 0 for no resampling.
    protect=0.33,
    version="v2")

def double_infer(input: str, model: str) -> str:
    output = f"{uuid4()}.wav"
    infer(input, output, model)
    infer(output, output, model)
    return output
# Call the main function which uses the arguments from sys.argv

if __name__ == '__main__':
    model = models_dict['Lexa']
    input_file = 'input.mp3'
    result = double_infer(input_file, model)
    print(result)