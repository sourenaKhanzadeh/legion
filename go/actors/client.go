package actors

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/go-resty/resty/v2"
)

type ComputeRequest struct {
	Data      []float32 `json:"data"`
	Operation string    `json:"operation"`
}

type ComputeResponse struct {
	Result []float32 `json:"result"`
	Error  string    `json:"error"`
}

func Compute(data []float32, operation string) ([]float32, error) {
	// Define the GPU server URL
	serverURL := "http://localhost:8001/compute"

	// Prepare the request payload
	requestData := ComputeRequest{
		Data:      []float32{1, 2, 3, 4},
		Operation: "cube",
	}

	// Create a new HTTP client
	client := resty.New()

	// Send the request
	var response ComputeResponse
	resp, err := client.R().
		SetBody(requestData).
		SetResult(&response).
		Post(serverURL)

	// Handle errors
	if err != nil {
		log.Fatalf("Request failed: %v", err)
	}

	// Check for server-side errors
	if response.Error != "" {
		log.Fatalf("Server error: %s", response.Error)
	}

	// Print the result
	fmt.Println("Response:", resp)
	fmt.Println("Computed Result:", response.Result)

	return response.Result, nil
}

func GetGPUs() ([]map[string]string, error) {
	// Define the master server URL
	serverURL := "http://localhost:8001/nvidia-smi"

	// Create a new HTTP client
	client := resty.New()

	// Get the raw string response first
	var rawResponse string
	resp, err := client.R().
		SetResult(&rawResponse).
		Get(serverURL)

	if err != nil {
		log.Fatalf("Request failed: %v", err)
	}

	// Parse the string into our desired format
	var gpus []map[string]string
	err = json.Unmarshal([]byte(rawResponse), &gpus)
	if err != nil {
		// Print the raw response to help debug
		fmt.Printf("Raw response: %s\n", rawResponse)
		return nil, fmt.Errorf("failed to unmarshal response: %v", err)
	}

	// Print the result
	fmt.Println("Response:", resp)
	fmt.Println("GPUs:", gpus)

	return gpus, nil
}
