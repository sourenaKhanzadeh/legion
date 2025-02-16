package actors

import (
	"errors"

	"github.com/go-resty/resty/v2"
)

type ExecRequest struct {
	ScriptPath string `json:"script_path"`
}

type ExecResponse struct {
	Output string `json:"output"`
	Error  string `json:"error"`
}

type ProjectRequest struct {
	ProjectZip []byte `json:"project_zip"`
}

type ProjectResponse struct {
	Output string `json:"output"`
	Error  string `json:"error"`
	Worker string `json:"worker"`
}

func Exec(scriptPath string) ([]string, error) {
	serverURL := "http://localhost:8001/execute_script"

	requestData := ExecRequest{
		ScriptPath: scriptPath,
	}

	client := resty.New()

	var response []ExecResponse
	_, err := client.R().
		SetBody(requestData).
		SetResult(&response).
		Post(serverURL)

	if err != nil {
		return nil, err
	}

	outputs := make([]string, len(response))
	for i, res := range response {
		outputs[i] = res.Output
	}

	return outputs, nil
}

func ExecProject(projectZip []byte) ([]string, error) {
	serverURL := "http://localhost:8001/execute_project"

	requestData := ProjectRequest{
		ProjectZip: projectZip,
	}

	client := resty.New()

	var response []ProjectResponse
	_, err := client.R().
		SetBody(requestData).
		SetResult(&response).
		Post(serverURL)

	if err != nil {
		return nil, err
	}

	var outputs []string
	for _, res := range response {
		if res.Error != "" {
			return nil, errors.New(res.Error)
		}
		outputs = append(outputs, res.Output)
	}
	return outputs, nil
}
