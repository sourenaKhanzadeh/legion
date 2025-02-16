package main

import (
	"Legion-Go/actors"
	"fmt"
	"log"
	"math/rand/v2"
)

func main() {
	fmt.Println("GETTING GPUS!")
	gpus, err := actors.GetGPUs()
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	fmt.Println("GPUs:", gpus)

	fmt.Println("COMPUTE!")
	result, err := actors.Compute([]float32{1, 2, 3, 4}, "square")
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	fmt.Println(result)

	fmt.Println("MATMUL!")
	// make two nxn matrices Randomly
	A := GenerateRandomMatrix(50, 50)
	B := GenerateRandomMatrix(50, 50)
	matResult, err := actors.MatMul(A, B)
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	fmt.Println(matResult)
}

// GenerateRandomMatrix creates an N x P matrix with random float values
func GenerateRandomMatrix(rows, cols int) [][]float32 {
	matrix := make([][]float32, rows)
	for i := range matrix {
		matrix[i] = make([]float32, cols)
		for j := range matrix[i] {
			matrix[i][j] = rand.Float32() * 10 // Random values between 0 and 10
		}
	}
	return matrix
}
