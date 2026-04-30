package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

const (
	ReadFunctionName = "Read"
	FilePathParam    = "file_path"
)

func main() {
	var prompt string
	flag.StringVar(&prompt, "p", "", "Prompt to send to LLM")
	flag.Parse()

	if prompt == "" {
		panic("Prompt must not be empty")
	}

	apiKey := os.Getenv("OPENROUTER_API_KEY")
	baseUrl := os.Getenv("OPENROUTER_BASE_URL")
	if baseUrl == "" {
		baseUrl = "https://openrouter.ai/api/v1"
	}

	if apiKey == "" {
		panic("Env variable OPENROUTER_API_KEY not found")
	}

	client := openai.NewClient(option.WithAPIKey(apiKey), option.WithBaseURL(baseUrl))
	resp, err := client.Chat.Completions.New(context.Background(),
		openai.ChatCompletionNewParams{
			Model: "anthropic/claude-haiku-4.5",
			Messages: []openai.ChatCompletionMessageParamUnion{
				{
					OfUser: &openai.ChatCompletionUserMessageParam{
						Content: openai.ChatCompletionUserMessageParamContentUnion{
							OfString: openai.String(prompt),
						},
					},
				},
			},
			Tools: []openai.ChatCompletionToolUnionParam{
				{
					OfFunction: &openai.ChatCompletionFunctionToolParam{
						Function: shared.FunctionDefinitionParam{
							Name:        "Read",
							Description: openai.String("Read and return the contents of a file"),
							Parameters: openai.FunctionParameters{
								"type": "object",
								"properties": map[string]any{
									"file_path": map[string]string{
										"type":        "string",
										"description": "The path to the file to read",
									},
								},
								"required": []string{"file_path"},
							},
						},
					},
				},
			},
		},
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	if len(resp.Choices) == 0 {
		panic("No choices in response")
	}

	// You can use print statements as follows for debugging, they'll be visible when running tests.
	fmt.Fprintln(os.Stderr, "Logs from your program will appear here!")

	hasToolCalls := len(resp.Choices[0].Message.ToolCalls) > 0
	if hasToolCalls {
		toolCall := resp.Choices[0].Message.ToolCalls[0]
		funcName := toolCall.Function.Name

		if !hasValidateArgs(toolCall.Function.Arguments) {
			fmt.Fprintf(os.Stderr, "Arguments for function %s are not valid\n", funcName)
		}

		switch funcName {
		case ReadFunctionName:
			readFile(toolCall.Function.Arguments)
		}

		return
	}

	fmt.Print(resp.Choices[0].Message.Content)
}

func hasValidateArgs(args string) bool {
	return json.Valid([]byte(args))
}

func readFile(args string) {
	var jsonFields map[string]any
	if err := json.Unmarshal([]byte(args), &jsonFields); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing arguments: %v\n", err)
		return
	}

	filePath, ok := jsonFields[FilePathParam]
	if !ok {
		fmt.Fprintf(os.Stderr, "Missing required argument: %s\n", FilePathParam)
		return
	}

	fileContent, err := os.ReadFile(filePath.(string))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
		return
	}

	fmt.Print(string(fileContent))
}
