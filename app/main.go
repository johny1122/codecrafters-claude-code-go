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
	ReadFunctionName  = "Read"
	WriteFunctionName = "Write"
	FilePathParam     = "file_path"
	ContentParam      = "content"
	StopReason        = "stop"
	RoleTool          = "tool"
)

var (
	readTool = openai.ChatCompletionToolUnionParam{
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
	}

	writeTool = openai.ChatCompletionToolUnionParam{
		OfFunction: &openai.ChatCompletionFunctionToolParam{
			Function: shared.FunctionDefinitionParam{
				Name:        "Write",
				Description: openai.String("Write content to a file"),
				Parameters: openai.FunctionParameters{
					"type":     "object",
					"required": []string{"file_path", "content"},
					"properties": map[string]any{
						"file_path": map[string]string{
							"type":        "string",
							"description": "The path to the file to write to",
						},
						"content": map[string]string{
							"type":        "string",
							"description": "The content to write to the file",
						},
					},
				},
			},
		},
	}
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

	messages := []openai.ChatCompletionMessageParamUnion{
		{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String(prompt),
				},
			},
		},
	}

	for {
		resp, err := client.Chat.Completions.New(context.Background(),
			openai.ChatCompletionNewParams{
				Model:    "anthropic/claude-haiku-4.5",
				Messages: messages,
				Tools:    []openai.ChatCompletionToolUnionParam{readTool, writeTool},
			},
		)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		if len(resp.Choices) == 0 {
			panic("No choices in response")
		}

		if resp.Choices[0].FinishReason == StopReason {
			fmt.Print(resp.Choices[0].Message.Content)
			os.Exit(0)
		}

		messages = append(messages, buildAssistantMsg(resp))

		// You can use print statements as follows for debugging, they'll be visible when running tests.
		fmt.Fprintln(os.Stderr, "Logs from your program will appear here!")

		for _, toolCall := range resp.Choices[0].Message.ToolCalls {
			content, err := executeTool(toolCall)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error executing tool: %v\n", err)
				return
			}

			messages = append(messages, buildToolMessage(toolCall.ID, content))
		}
	}
}

func buildAssistantMsg(resp *openai.ChatCompletion) openai.ChatCompletionMessageParamUnion {
	assistantMsg := openai.ChatCompletionMessageParamUnion{
		OfAssistant: &openai.ChatCompletionAssistantMessageParam{
			Content: openai.ChatCompletionAssistantMessageParamContentUnion{
				OfString: openai.String(resp.Choices[0].Message.Content),
			},
		},
	}

	hasToolCalls := len(resp.Choices[0].Message.ToolCalls) > 0
	if hasToolCalls {
		assistantMsg.OfAssistant.ToolCalls = []openai.ChatCompletionMessageToolCallUnionParam{
			{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: resp.Choices[0].Message.ToolCalls[0].ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      resp.Choices[0].Message.ToolCalls[0].Function.Name,
						Arguments: resp.Choices[0].Message.ToolCalls[0].Function.Arguments,
					},
				},
			},
		}
	}
	return assistantMsg
}

func buildToolMessage(toolCallID, content string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{
		OfTool: &openai.ChatCompletionToolMessageParam{
			ToolCallID: toolCallID,
			Content: openai.ChatCompletionToolMessageParamContentUnion{
				OfString: openai.String(content),
			},
		},
	}
}

func executeTool(toolCall openai.ChatCompletionMessageToolCallUnion) (string, error) {
	funcName := toolCall.Function.Name

	if !toolCall.Function.JSON.Arguments.Valid() {
		return "", fmt.Errorf("invalid arguments for function %s", funcName)
	}

	switch funcName {
	case ReadFunctionName:
		content, err := readFile(toolCall.Function.Arguments)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
			return "", err
		}
		return content, nil
	case WriteFunctionName:
		content, err := writeFile(toolCall.Function.Arguments)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error writing file: %v\n", err)
			return "", err
		}
		return content, nil
	}

	return "", fmt.Errorf("unknown function: %s", funcName)
}

func writeFile(args string) (string, error) {
	var jsonFields map[string]any
	if err := json.Unmarshal([]byte(args), &jsonFields); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing arguments: %v\n", err)
		return "", err
	}

	filePath, ok := jsonFields[FilePathParam]
	if !ok {
		fmt.Fprintf(os.Stderr, "Missing required argument: %s\n", FilePathParam)
		return "", fmt.Errorf("missing required argument: %s", FilePathParam)
	}

	content, ok := jsonFields[ContentParam]
	if !ok {
		fmt.Fprintf(os.Stderr, "Missing required argument: %s\n", ContentParam)
		return "", fmt.Errorf("missing required argument: %s", ContentParam)
	}

	err := os.WriteFile(filePath.(string), []byte(content.(string)), 0644)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error writing file: %v\n", err)
		return "", err
	}
	return "wrote content to file successfully", nil
}

func readFile(args string) (string, error) {
	var jsonFields map[string]any
	if err := json.Unmarshal([]byte(args), &jsonFields); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing arguments: %v\n", err)
		return "", err
	}

	filePath, ok := jsonFields[FilePathParam]
	if !ok {
		fmt.Fprintf(os.Stderr, "Missing required argument: %s\n", FilePathParam)
		return "", fmt.Errorf("missing required argument: %s", FilePathParam)
	}

	fileContent, err := os.ReadFile(filePath.(string))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
		return "", err
	}

	return string(fileContent), nil
}
