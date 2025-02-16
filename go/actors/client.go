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
	// Define the Master Server URL (not a worker)
	serverURL := "http://localhost:8001/compute" // Fix: Use Master Server

	// Prepare the request payload
	requestData := ComputeRequest{
		Data:      data,      // Fix: Use function parameters
		Operation: operation, // Fix: Use function parameters
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
	// Define the Master Server URL for GPU details
	serverURL := "http://localhost:8001/nvidia-smi" // Fix: Use Master Server

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

type MatMulRequest struct {
	A [][]float32 `json:"A"`
	B [][]float32 `json:"B"`
}

type MatMulResponse struct {
	Result [][]float32 `json:"result"`
	Error  string      `json:"error"`
}

func MatMul(A [][]float32, B [][]float32) ([][]float32, error) {
	// Master Server URL
	serverURL := "http://localhost:8001/matmul"

	// Prepare the request payload
	requestData := MatMulRequest{
		A: A,
		B: B,
	}

	// Create a new HTTP client
	client := resty.New()

	// Send the request
	var response MatMulResponse
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
	fmt.Println("Computed Matrix:", response.Result)

	return response.Result, nil
}
