import gradio as gr
import subprocess
import re
import threading

# Global variable to store the subprocess
process = None
stop_flag = threading.Event()  # To signal stopping the process

def run_finetuning_with_progress(model, dataset, instruction, input_text, output_text, promptType, max_steps, save_steps, output_dir):
    global process
    stop_flag.clear()  # Reset the stop flag

    # Fine-tuning command
    command = [
        "python", "qlora_finetuning.py",
        "--repo-id-or-model-path", model,
        "--dataset", dataset,
        "--instruction", instruction,
        "--input", input_text,
        "--output", output_text,
        "--prompt_type", promptType,
        "--max_steps", max_steps,
        "--save_steps", save_steps,
        "--output_dir", output_dir
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    total_steps = None  # Will be determined dynamically if available
    for line in iter(process.stdout.readline, ""):
        if stop_flag.is_set():
            process.terminate()
            yield {"logs": "Fine-tuning stopped by the user.", "progress": 0}
            break

        # Send logs to Gradio frontend
        yield {"logs": line.strip(), "progress": None}

        # Parse progress from the tqdm-like output
        match = re.search(r'(\d+)/(\d+)', line)
        if match:
            current_step, total_steps = map(int, match.groups())
            progress = int((current_step / total_steps) * 100)
            yield {"logs": None, "progress": progress}  # Send progress updates

    if process and process.poll() is None:  # Ensure the process is terminated if running
        process.terminate()

    if not stop_flag.is_set():  # Finalize if not stopped by the user
        if process.returncode == 0:
            yield {"logs": "Fine-tuning complete!", "progress": 100}
            # Run the second script
            yield from run_adapter_script(output_dir, max_steps)
        else:
            yield {"logs": "An error occurred during fine-tuning.", "progress": 0}

def run_adapter_script(output_dir, max_steps):
    """Run the second script after fine-tuning."""
    adapter_path = f"./outputs/{output_dir}/checkpoint-{max_steps}"
    output_path = f"./outputs/{output_dir}/checkpoint-{max_steps}-merged"

    command = [
        "python", "export_merged_model.py",  # Replace with your script name
        "--adapter_path", adapter_path,
        "--output_path", output_path
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(process.stdout.readline, ""):
            yield {"logs": line.strip(), "progress": None}
        process.wait()
        if process.returncode == 0:
            yield {"logs": "Adapter merging complete!", "progress": 100}
        else:
            yield {"logs": "An error occurred during adapter merging.", "progress": 0}
    except Exception as e:
        yield {"logs": f"Error running adapter script: {str(e)}", "progress": 0}


def stop_finetuning():
    stop_flag.set()  # Signal to stop the process
    if process and process.poll() is None:  # Check if the process is running
        process.terminate()
    return "Fine-tuning stopped."

# Define Gradio components
with gr.Blocks() as app:
    gr.Markdown("# QLoRA Fine-Tuning with Progress and Stop Button")

    model = gr.Textbox(label="Model Path", placeholder="Enter the model path")
    dataset = gr.Textbox(label="Dataset Path", placeholder="Enter the dataset path")
    instruction = gr.Textbox(label="Instruction", placeholder="Enter the instruction text")
    input_text = gr.Textbox(label="Input", placeholder="Enter the input text")
    output_text = gr.Textbox(label="Output", placeholder="Enter the output text")
    promptType = gr.Textbox(label="Prompt Type", placeholder="Enter the prompt type")
    max_steps = gr.Textbox(label="Max Steps", value=100, interactive=True)
    save_steps = gr.Textbox(label="Save Steps", value=100, interactive=True)
    output_dir = gr.Textbox(label="Output Directory", placeholder="Enter the output directory path")
    start_button = gr.Button("Start Fine-Tuning")
    stop_button = gr.Button("Stop Fine-Tuning")
    
    logs = gr.Textbox(label="Logs", interactive=False)
    progress_bar = gr.Slider(label="Progress", value=0, minimum=0, maximum=100, interactive=False)  # Slider as a progress bar
    
    # Event handler for fine-tuning
    def fine_tuning_handler(model, dataset, instruction, input_text, output_text, promptType, max_steps, save_steps, output_dir):
        for result in run_finetuning_with_progress(model, dataset, instruction, input_text, output_text, promptType, max_steps, save_steps, output_dir):
            yield gr.update(value=result["logs"]), gr.update(value=result["progress"])

    # Connect event handlers to buttons
    start_button.click(
        fn=fine_tuning_handler,
        inputs=[model, dataset, instruction, input_text, output_text, promptType, max_steps, save_steps, output_dir],
        outputs=[logs, progress_bar]
    )
    stop_button.click(
        fn=stop_finetuning,
        inputs=[],
        outputs=[logs]
    )

app.launch()
