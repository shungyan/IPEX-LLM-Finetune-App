package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"net/http"
	"os/exec"

	"github.com/gin-gonic/gin"
)

// Global variable to store logs
var logsChannel = make(chan string)

func main() {
	r := gin.Default()

	// Serve static HTML from the templates directory
	r.LoadHTMLGlob("templates/*")

	// Route to render the HTML form
	r.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", nil)
	})

	// Handle the form submission
	r.POST("/finetune", func(c *gin.Context) {
		model := c.PostForm("model")
		dataset := c.PostForm("dataset")
		instruction := c.PostForm("instruction")
		input := c.PostForm("input") // This can be empty for prompt_no_input
		output := c.PostForm("output")
		promptType := c.PostForm("promptType")
		maxsteps := c.PostForm("maxsteps")
		savesteps := c.PostForm("savesteps")
		outputdir := c.PostForm("outputdir") // Add this line

		if model == "" || dataset == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Model and dataset are required"})
			return
		}

		// Update function call to include savesteps
		go runFineTuneProcess(model, dataset, instruction, input, output, promptType, maxsteps, savesteps, outputdir)

		c.JSON(http.StatusOK, gin.H{
			"message":     "Fine-tuning started",
			"model":       model,
			"dataset":     dataset,
			"instruction": instruction,
			"input":       input,
			"output":      output,
			"promptType":  promptType,
			"maxsteps":    maxsteps,
			"savesteps":   savesteps,
			"outputdir":   outputdir, // Add this line
		})
	})

	// Route to stream logs back to the frontend
	r.GET("/logs", func(c *gin.Context) {
		// Set SSE headers for real-time updates
		c.Writer.Header().Set("Content-Type", "text/event-stream")
		c.Writer.Header().Set("Cache-Control", "no-cache")
		c.Writer.Header().Set("Connection", "keep-alive")

		// Flush logs to the client in real-time
		for logMessage := range logsChannel {
			c.SSEvent("message", logMessage)
			c.Writer.Flush()
		}
	})

	// Handle the post-processing request
	r.POST("/postprocess", func(c *gin.Context) {
		// Define a struct to hold the 'maxsteps' field
		var jsonData struct {
			maxsteps string `json:"maxsteps" binding:"required"`
		}

		// Bind the JSON body to the struct (only binds the 'maxsteps' field)
		if err := c.ShouldBindJSON(&jsonData); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Call the post-processing function asynchronously with just the maxsteps value
		go runPostProcess(jsonData.maxsteps)

		c.JSON(http.StatusOK, gin.H{
			"message": "Post-processing started",
		})
	})

	// Start the server on port 12345
	r.Run(":12345")
}

func runPostProcess(maxsteps, outputdir string) {

	outputPath := fmt.Sprintf("./outputs/%s/checkpoint-%s-merged", outputdir, maxsteps)

	cmd := exec.Command("python3", "../LLM-Finetuning/QLoRA/simple-example/export_merged_model.py", "--repo-id-or-model-path", "meta-llama/Llama-2-7b-hf", "--adapter_path", outputdir, "--output_path", outputPath)

	// Capture stdout and stderr of the process
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatalf("Error obtaining stdout: %v", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Fatalf("Error obtaining stderr: %v", err)
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		log.Fatalf("Failed to start the command: %v", err)
	}

	// Stream logs from stdout
	go streamLogs(stdout)
	go streamLogs(stderr)

	// Wait for the command to finish
	if err := cmd.Wait(); err != nil {
		log.Fatalf("post process exited with error: %v", err)
	}

	log.Printf("merge completed")

	ggufPath := fmt.Sprintf("./outputs/%s-%s.gguf", outputdir, maxsteps)

	cmd2 := exec.Command("python3", "../llama.cpp/convert_hf_to_gguf.py", outputPath, "--outfile", ggufPath)

	// Capture stdout and stderr of the process
	stdout2, err := cmd2.StdoutPipe()
	if err != nil {
		log.Fatalf("Error obtaining stdout: %v", err)
	}
	stderr2, err := cmd2.StderrPipe()
	if err != nil {
		log.Fatalf("Error obtaining stderr: %v", err)
	}

	// Start the command
	if err := cmd2.Start(); err != nil {
		log.Fatalf("Failed to start the command: %v", err)
	}

	// Stream logs from stdout
	go streamLogs(stdout2)
	go streamLogs(stderr2)

	// Wait for the command to finish
	if err := cmd2.Wait(); err != nil {
		log.Fatalf("post process exited with error: %v", err)
	}

	log.Printf("GGUF conversion completed")
}

// Backend process to run the fine-tuning script and capture logs
func runFineTuneProcess(model, dataset, instruction, input, output, promptType, maxsteps, savesteps, outputdir string) {
	log.Printf("Starting fine-tuning with model: %s, dataset: %s\n", model, dataset)

	// Call the Python script with model and dataset as arguments
	cmd := exec.Command("python3",
		"../LLM-Finetuning/QLoRA/simple-example/qlora_finetuning.py",
		"--repo-id-or-model-path", model,
		"--dataset", dataset,
		"--prompt_type", promptType,
		"--instruction", instruction,
		"--input", input,
		"--output", output,
		"--max_steps", maxsteps,
		"--save_steps", savesteps,
		"--outputdir", outputdir) // Add this line

	// Capture stdout and stderr of the process
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatalf("Error obtaining stdout: %v", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Fatalf("Error obtaining stderr: %v", err)
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		log.Fatalf("Failed to start the command: %v", err)
	}

	// Stream logs from stdout
	go streamLogs(stdout)
	go streamLogs(stderr)

	// Wait for the command to finish
	if err := cmd.Wait(); err != nil {
		log.Fatalf("Fine-tuning process exited with error: %v", err)
	}

	log.Printf("Fine-tuning process completed for model: %s, dataset: %s\n", model, dataset)
}

// Function to stream logs from a pipe
func streamLogs(pipe io.ReadCloser) {
	scanner := bufio.NewScanner(pipe)
	for scanner.Scan() {
		logMessage := scanner.Text()
		logsChannel <- logMessage // Send log message to the logsChannel
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading logs: %v", err)
	}
}
