package main

import (
	"Legion-Go/actors"
	"fmt"
	"log"
)

func main() {
	fmt.Println("COMPUTE!")
	result, err := actors.Compute([]float32{1, 2, 3, 4}, "square")
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	fmt.Println(result)
}
