package actors

import (
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
	serverURL := "http://localhost:8000/compute"

	// Prepare the request payload
	requestData := ComputeRequest{
		Data:      []float32{1, 2, 3, 4},
		Operation: "square",
	}

	// Create a new HTTP client
	client := resty.New()

	// Send the request
	var response ComputeResponse
	_, err := client.R().
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
	fmt.Println("Computed Result:", response.Result)

	return response.Result, nil
}
