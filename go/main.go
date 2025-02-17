package main

import (
	"Legion-Go/actors"
	"archive/zip"
	"bytes"
	"fmt"
	"io"
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

	// Save zip file
	err = os.WriteFile("project.zip", projectZip, 0644)
	if err != nil {
		log.Fatalf("Error saving zip: %v", err)
	}

	absPath, err := filepath.Abs("project.zip")
	if err != nil {
		log.Fatalf("Error getting absolute path: %v", err)
	}
	projectResult, err := actors.ExecProject(absPath)
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
	// Create a buffer to store the ZIP file
	var buf bytes.Buffer
	zipWriter := zip.NewWriter(&buf)

	// Walk through all files & subdirectories
	err := filepath.Walk(projectPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories, we only need files
		if info.IsDir() {
			return nil
		}

		// Open file
		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()

		// Create a ZIP entry with relative path
		relPath, err := filepath.Rel(projectPath, path) // ✅ Fix: Use relative path
		if err != nil {
			return err
		}

		zipEntry, err := zipWriter.Create(relPath)
		if err != nil {
			return err
		}

		// Copy file content into ZIP
		_, err = io.Copy(zipEntry, file)
		if err != nil {
			return err
		}

		return nil
	})

	// Close ZIP writer
	zipWriter.Close()

	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}
