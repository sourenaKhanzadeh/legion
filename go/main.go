package main

import (
	"Legion-Go/actors"
	"fmt"
	"log"
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
	matResult, err := actors.MatMul([][]float32{{1, 2}, {3, 4}}, [][]float32{{5, 6}, {7, 8}})
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	fmt.Println(matResult)
}
