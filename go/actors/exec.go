package actors

import (
	"github.com/go-resty/resty/v2"
)

type ExecRequest struct {
	ScriptPath string `json:"script_path"`
}

type ExecResponse struct {
	Output string `json:"output"`
}

func Exec(scriptPath string) (string, error) {
	serverURL := "http://localhost:8001/execute_script"

	requestData := ExecRequest{
		ScriptPath: scriptPath,
	}

	client := resty.New()

	var response ExecResponse
	_, err := client.R().
		SetBody(requestData).
		SetResult(&response).
		Post(serverURL)

	if err != nil {
		return "", err
	}

	return response.Output, nil
}
