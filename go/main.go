package main

import (
	"Legion-Go/actors"
	"archive/zip"
	"bytes"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"path/filepath"
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
	A := GenerateRandomMatrix(2, 2)
	B := GenerateRandomMatrix(2, 2)
	matResult, err := actors.MatMul(A, B)
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	fmt.Println(matResult)

	// fmt.Println("EXEC!")
	// execResult, err := actors.Exec("../python/scripts/exec_test.py")
	// if err != nil {
	// 	log.Fatalf("Error: %v", err)
	// }
	// fmt.Println(execResult)

	fmt.Println("EXEC PROJECT!")
	projectZip, err := zipProject("../proj/Moin")
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	projectResult, err := actors.ExecProject(projectZip)
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	fmt.Println(projectResult)
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

func zipProject(projectPath string) ([]byte, error) {
	// Create a buffer to write the zip to
	var buf bytes.Buffer
	zipWriter := zip.NewWriter(&buf)
	defer zipWriter.Close()

	projectFiles, err := os.ReadDir(projectPath)
	if err != nil {
		return nil, err
	}

	for _, file := range projectFiles {
		filePath := filepath.Join(projectPath, file.Name())
		if file.IsDir() {
			continue
		}

		zipEntry, err := zipWriter.Create(filePath)
		if err != nil {
			return nil, err
		}

		content, err := os.ReadFile(filePath)
		if err != nil {
			return nil, err
		}

		_, err = zipEntry.Write(content)
		if err != nil {
			return nil, err
		}
	}

	zipWriter.Close()
	return buf.Bytes(), nil
}
