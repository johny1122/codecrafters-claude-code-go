package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"slices"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

const (
	ReadFunctionName            = "Read"
	WriteFunctionName           = "Write"
	BashFunctionName            = "Bash"
	FilePathParam               = "file_path"
	ContentParam                = "content"
	CommandParam                = "command"
	StopReason                  = "stop"
	RoleTool                    = "tool"
	SkillsFolderPath            = ".claude/skills"
	ScriptLocationFormatMessage = "Skill: %s (located at .claude/skills/%s)\nPaths in the instructions below are relative to that folder.\n"
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

	bashTool = openai.ChatCompletionToolUnionParam{
		OfFunction: &openai.ChatCompletionFunctionToolParam{
			Function: shared.FunctionDefinitionParam{
				Name:        "Bash",
				Description: openai.String("Execute a shell command"),
				Parameters: openai.FunctionParameters{
					"type":     "object",
					"required": []string{"command"},
					"properties": map[string]any{
						"command": map[string]string{
							"type":        "string",
							"description": "The command to execute",
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

	skillsMetadata, skillNames, err := getSkillsMetadata()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting skills metadata: %v\n", err)
		os.Exit(1)
	}

	messages := []openai.ChatCompletionMessageParamUnion{}
	if skillsMetadata != "" {
		messages = append(messages, openai.ChatCompletionMessageParamUnion{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String("You have access to the following skills: \n" + skillsMetadata),
				},
			},
		})
	}

	if strings.Index(prompt, "/") == 0 { // stack skills
		promptParts := strings.Fields(prompt)
		foundSkills := []string{}
		indexOfFisrtArgument := len(prompt)
		for _, skillName := range promptParts {
			if slices.Contains(skillNames, skillName[1:]) { // remove the leading slash
				foundSkills = append(foundSkills, skillName)
			} else {
				indexOfFisrtArgument = strings.Index(prompt, skillName)
				break
			}
		}

		for _, skillName := range foundSkills {
			message := replaceArgumentsInPrompt(skillName, skillName[1:], strings.Fields(prompt[indexOfFisrtArgument:]))
			message = addScriptsLocationToPrompt(message, skillName)
			messages = append(messages, openai.ChatCompletionMessageParamUnion{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String(message),
					},
				},
			})
		}

	} else {
		hasSlashCommand := strings.Contains(prompt, " /") || strings.HasPrefix(prompt, "/")
		if hasSlashCommand {
			prompt = getPromptWithSlashCommand(prompt)
		}

		messages = append(messages, openai.ChatCompletionMessageParamUnion{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String(prompt),
				},
			},
		})
	}

	for {
		resp, err := client.Chat.Completions.New(context.Background(),
			openai.ChatCompletionNewParams{
				Model:    "anthropic/claude-haiku-4.5",
				Messages: messages,
				Tools:    []openai.ChatCompletionToolUnionParam{readTool, writeTool, bashTool},
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

func getPromptWithSlashCommand(prompt string) string {
	slashIndex := strings.Index(prompt, "/")
	skillName := ""
	spaceIndex := strings.Index(prompt[slashIndex:], " ")
	if spaceIndex != -1 {
		skillName = prompt[slashIndex+1 : slashIndex+spaceIndex]
	} else {
		skillName = prompt[slashIndex+1:]
	}

	arguments := getArgumentsFromPrompt(prompt, skillName, slashIndex)
	prompt = replaceArgumentsInPrompt(prompt, skillName, arguments)
	prompt = addScriptsLocationToPrompt(prompt, skillName)

	return prompt
}

func addScriptsLocationToPrompt(prompt, skillName string) string {
	_, err := os.ReadDir(SkillsFolderPath + "/" + skillName + "/scripts")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading skills directory: %v\n", err)
		return prompt
	}

	prompt = fmt.Sprintf(ScriptLocationFormatMessage, skillName, skillName) + prompt

	return prompt

}

func replaceArgumentsInPrompt(prompt, skillName string, arguments []string) string {
	prompt = strings.Replace(prompt, "/"+skillName, getSkillBody(skillName), 1)
	prompt = strings.Replace(prompt, "$ARGUMENTS", strings.Join(arguments, " "), 1)
	for i, arg := range arguments {
		prompt = strings.Replace(prompt, fmt.Sprintf("$%d", i), arg, 1)
	}

	return prompt
}

func getArgumentsFromPrompt(prompt, skillName string, slashIndex int) []string {
	return strings.Fields(prompt[slashIndex+1+len(skillName):])
}

func getSkillBody(skillName string) string {
	skillFilePath := fmt.Sprintf("%s/%s/SKILL.md", SkillsFolderPath, skillName)
	content, err := os.ReadFile(skillFilePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading skill file %s: %v\n", skillFilePath, err)
		return ""
	}

	lines := strings.Split(string(content), "\n")
	frontmatterFound := 0
	skillBody := ""
	for _, line := range lines {
		if frontmatterFound == 2 {
			skillBody += line + "\n"
			continue
		}

		if strings.TrimSpace(line) == "---" {
			frontmatterFound++
		}
	}

	return skillBody
}

func getSkillsMetadata() (string, []string, error) {
	files, err := os.ReadDir(SkillsFolderPath)
	if err != nil {
		return "", nil, nil
	}

	var skills []string
	var skillNames []string
	for _, file := range files {
		if file.IsDir() {
			skillFilePath := fmt.Sprintf("%s/%s/SKILL.md", SkillsFolderPath, file.Name())
			content, err := os.ReadFile(skillFilePath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error reading skill file %s: %v\n", skillFilePath, err)
				continue
			}

			lines := strings.Split(string(content), "\n")
			frontmatterFound := 0
			name, description := "", ""
			inDescription := false
			for _, line := range lines {
				if strings.TrimSpace(line) == "---" {
					frontmatterFound++
					if frontmatterFound == 2 {
						break
					}
				} else if inDescription {
					description += " " + line
					continue
				} else if strings.HasPrefix(line, "name: ") {
					name = strings.TrimSpace(line[5:])
				} else if strings.HasPrefix(line, "description: ") {
					description = strings.TrimSpace(line[13:])
					inDescription = true
				}
			}
			skillNames = append(skillNames, name)
			skills = append(skills, fmt.Sprintf("%s: %s\n", name, description))
		}
	}

	return strings.Join(skills, "\n"), skillNames, nil
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
	case BashFunctionName:
		content, err := executeBashCommand(toolCall.Function.Arguments)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error executing bash command: %v\n", err)
			return "", err
		}
		return content, nil
	}

	return "", fmt.Errorf("unknown function: %s", funcName)
}

func executeBashCommand(args string) (string, error) {
	var jsonFields map[string]any
	if err := json.Unmarshal([]byte(args), &jsonFields); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing arguments: %v\n", err)
		return "", err
	}

	command, ok := jsonFields[CommandParam]
	if !ok {
		fmt.Fprintf(os.Stderr, "Missing required argument: %s\n", CommandParam)
		return "", fmt.Errorf("missing required argument: %s", CommandParam)
	}

	commandStr := command.(string)
	commandParts := strings.Split(commandStr, " ")
	fmt.Fprintln(os.Stderr, "Executing command:", commandStr)

	if len(commandParts) == 0 {
		return "", fmt.Errorf("given command is empty")
	}

	cmd := exec.Command(commandParts[0])
	if len(commandParts) > 1 {
		cmd = exec.Command(commandParts[0], commandParts[1:]...)
	}

	output, err := cmd.CombinedOutput()
	fmt.Fprintln(os.Stderr, "Command output:", string(output))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error executing bash command: %v\n", err)
		return err.Error(), nil
	}
	return string(output), nil
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
