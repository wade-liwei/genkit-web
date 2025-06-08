// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
	"github.com/rs/zerolog/log"
)

// HeaderContextKey 用于存储 http.Header
type HeaderContextKey struct{}

// WithHeaders 将 header 存入 context
func WithHeaders(ctx context.Context, headers http.Header) context.Context {
    return context.WithValue(ctx, HeaderContextKey{}, headers)
}

// GetHeaders 提取 header
func GetHeaders(ctx context.Context) http.Header {
    if headers, ok := ctx.Value(HeaderContextKey{}).(http.Header); ok {
        return headers
    }
    return nil
}

const simpleGreetingPromptTemplate = `
You're a barista at a nice coffee shop.
A regular customer named {{customerName}} enters.
Greet the customer in one sentence, and recommend a coffee drink.
`

const greetingWithHistoryPromptTemplate = `
{{role "user"}}
Hi, my name is {{customerName}}. The time is {{currentTime}}. Who are you?

{{role "model"}}
I am Barb, a barista at this nice underwater-themed coffee shop called Krabby Kooffee.
I know pretty much everything there is to know about coffee,
and I can cheerfully recommend delicious coffee drinks to you based on whatever you like.

{{role "user"}}
Great. Last time I had {{previousOrder}}.
I want you to greet me in one sentence, and recommend a drink.
`

type simpleGreetingInput struct {
    CustomerName string `json:"customerName"`
}

type customerTimeAndHistoryInput struct {
    CustomerName  string `json:"customerName"`
    CurrentTime   string `json:"currentTime"`
    PreviousOrder string `json:"previousOrder"`
}

type testAllCoffeeFlowsOutput struct {
    Pass    bool     `json:"pass"`
    Replies []string `json:"replies,omitempty"`
    Error   string   `json:"error,omitempty"`
}

func main() {
    ctx := context.Background()
    g, err := genkit.Init(ctx,
        genkit.WithDefaultModel("googleai/gemini-2.0-flash"),
        genkit.WithPlugins(&googlegenai.GoogleAI{}),
    )
    if err != nil {
        log.Fatal().Msgf("failed to create Genkit: %v", err)
    }

    m := googlegenai.GoogleAIModel(g, "gemini-2.0-flash")

    // 定义 simpleGreeting Prompt 和 Flow
    simpleGreetingPrompt, err := genkit.DefinePrompt(g, "simpleGreeting",
        ai.WithPrompt(simpleGreetingPromptTemplate),
        ai.WithModel(m),
        ai.WithInputType(simpleGreetingInput{}),
        ai.WithOutputFormat(ai.OutputFormatText),
    )
    if err != nil {
        log.Fatal().Msg(err.Error())
    }

    simpleGreetingFlow := genkit.DefineStreamingFlow(g, "simpleGreeting", func(ctx context.Context, input *simpleGreetingInput, cb func(context.Context, string) error) (string, error) {
        // 提取 HTTP header
        headers := GetHeaders(ctx)
        if headers != nil {
            log.Info().
                Str("authorization", headers.Get("Authorization")).
                Str("x-request-id", headers.Get("X-Request-ID")).
                Msg("Received HTTP headers in simpleGreeting")
        }

        // 验证 header
        if headers != nil && !strings.HasPrefix(headers.Get("Authorization"), "Bearer ") {
            return "", fmt.Errorf("invalid authorization header")
        }

        // 日志输入
        inputJSON, err := json.Marshal(input)
        if err != nil {
            return "", fmt.Errorf("json.Marshal: %w", err)
        }
        log.Info().Msgf("input--------%s", string(inputJSON))

        // 执行 prompt
        var callback func(context.Context, *ai.ModelResponseChunk) error
        if cb != nil {
            callback = func(ctx context.Context, c *ai.ModelResponseChunk) error {
                return cb(ctx, c.Text())
            }
        }
        resp, err := simpleGreetingPrompt.Execute(ctx,
            ai.WithInput(input),
            ai.WithStreaming(callback),
        )
        if err != nil {
            return "", err
        }
        return resp.Text(), nil
    })

    // 定义 greetingWithHistory Prompt 和 Flow
    greetingWithHistoryPrompt, err := genkit.DefinePrompt(g, "greetingWithHistory",
        ai.WithPrompt(greetingWithHistoryPromptTemplate),
        ai.WithModel(m),
        ai.WithInputType(customerTimeAndHistoryInput{}),
        ai.WithOutputFormat(ai.OutputFormatText),
    )
    if err != nil {
        log.Fatal().Msg(err.Error())
    }

    greetingWithHistoryFlow := genkit.DefineFlow(g, "greetingWithHistory", func(ctx context.Context, input *customerTimeAndHistoryInput) (string, error) {
        // 提取 HTTP header
        headers := GetHeaders(ctx)
        if headers != nil {
            log.Info().
                Str("authorization", headers.Get("Authorization")).
                Str("x-request-id", headers.Get("X-Request-ID")).
                Msg("Received HTTP headers in greetingWithHistory")
        }

        // 验证 header
        if headers != nil && !strings.HasPrefix(headers.Get("Authorization"), "Bearer ") {
            return "", fmt.Errorf("invalid authorization header")
        }

        // 日志输入
        inputJSON, err := json.Marshal(input)
        if err != nil {
            return "", fmt.Errorf("json.Marshal: %w", err)
        }
        log.Info().Msgf("input--------%s", string(inputJSON))

        // 执行 prompt
        resp, err := greetingWithHistoryPrompt.Execute(ctx,
            ai.WithInput(input),
        )
        if err != nil {
            return "", err
        }
        return resp.Text(), nil
    })

    // 定义 testAllCoffeeFlows Flow
    coffeeFlow := genkit.DefineFlow(g, "testAllCoffeeFlows", func(ctx context.Context, _ struct{}) (*testAllCoffeeFlowsOutput, error) {
        // 提取 HTTP header
        headers := GetHeaders(ctx)
        if headers != nil {
            log.Info().
                Str("authorization", headers.Get("Authorization")).
                Str("x-request-id", headers.Get("X-Request-ID")).
                Msg("Received HTTP headers in testAllCoffeeFlows")
        }



        // 验证 header
        if headers != nil && !strings.HasPrefix(headers.Get("Authorization"), "Bearer ") {
            return &testAllCoffeeFlowsOutput{
                Pass:  false,
                Error: "invalid authorization header",
            }, nil
        }


        headersAsJson, _ :=json.Marshal(headers)
        fmt.Println("headers-------------   ",string(headersAsJson))

        // 运行 simpleGreetingFlow
        test1, err := simpleGreetingFlow.Run(ctx, &simpleGreetingInput{
            CustomerName: "Sam",
        })
        if err != nil {
            return &testAllCoffeeFlowsOutput{
                Pass:  false,
                Error: err.Error(),
            }, nil
        }

        // 运行 greetingWithHistoryFlow
        test2, err := greetingWithHistoryFlow.Run(ctx, &customerTimeAndHistoryInput{
            CustomerName:  "Sam",
            CurrentTime:   "09:45am",
            PreviousOrder: "Caramel Macchiato",
        })
        if err != nil {
            return &testAllCoffeeFlowsOutput{
                Pass:  false,
                Error: err.Error(),
            }, nil
        }

        return &testAllCoffeeFlowsOutput{
            Pass: true,
            Replies: []string{
                test1,
                test2,
            },
        }, nil
    })

    // 自定义 ServeMux 和 Handler
    mux := http.NewServeMux()
    mux.HandleFunc("POST /testAllCoffeeFlows", func(w http.ResponseWriter, r *http.Request) {
        // 验证方法
        if r.Method != http.MethodPost {
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
            return
        }

        // 解析请求 body（允许空 JSON）
        var input struct{}
        if r.ContentLength > 0 {
            if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
                log.Error().Err(err).Msg("Failed to decode request body")
                http.Error(w, "Invalid input", http.StatusBadRequest)
                return
            }
        }

         headers, _ :=json.Marshal(r.Header)

         fmt.Println("headers-------------   ",string(headers))


        // 注入 header
        ctx := WithHeaders(r.Context(), r.Header)

        f := coffeeFlow

        // 调用 Flow
        output, err := f.Run(ctx, input)
        if err != nil {
            log.Error().Err(err).Msg("Flow execution failed")
            http.Error(w, fmt.Sprintf("Flow error: %v", err), http.StatusInternalServerError)
            return
        }

        // 返回响应
        w.Header().Set("Content-Type", "application/json")
        w.WriteHeader(http.StatusOK)
        if err := json.NewEncoder(w).Encode(output); err != nil {
            log.Error().Err(err).Msg("Failed to encode response")
        }
    })

    if err := server.Start(ctx, "127.0.0.1:8000", mux);err != nil{
    // 启动服务器
    log.Fatal().Msg(err.Error())
    }
}
