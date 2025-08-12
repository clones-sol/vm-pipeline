import { OpenAI } from 'openai';
import path from 'path';

interface MetaData {
  id: string;
  timestamp: string;
  duration_seconds: number;
  status: string;
  reason: string;
  title: string;
  description: string;
  platform: string;
  arch: string;
  version: string;
  locale: string;
  primary_monitor: {
    width: number;
    height: number;
  };
  quest: {
    title: string;
    app: string;
    icon_url: string;
    objectives: string[];
    content: string;
  };
}

interface Message {
  role: string;
  content: string | {
    type: string;
    data: string;
  };
}

interface EvaluationCriteria {
  outcomeAchievement: {
    weight: number;
    description: string;
  };
  processQuality: {
    weight: number;
    description: string;
  };
  efficiency: {
    weight: number;
    description: string;
  };
}

interface ChunkEvaluation {
  summary: string;
}

interface FinalEvaluation {
  summary: string;
  analysis: string;
  score: number;
  reasoning: string;
  confidence: number;
  outcomeAchievement: number;
  processQuality: number;
  efficiency: number;
}

interface GradeResult {
  summary: string;
  analysis: string;
  score: number;
  reasoning: string;
  confidence: number;
  outcomeAchievement: number;
  processQuality: number;
  efficiency: number;
}

export class Grader {
  private client: OpenAI;
  private chunkSize: number;
  private model: string;
  private evaluationCriteria: EvaluationCriteria;

  constructor(apiKey: string, chunkSize: number = 4, model?: string) {
    if (!apiKey) {
      throw new Error('OpenAI API key is required');
    }
    this.client = new OpenAI({ apiKey });
    this.chunkSize = chunkSize;
    // Use environment variable GRADER_MODEL if available, otherwise use provided model or default to gpt-4o
    this.model = model || process.env.GRADER_MODEL || 'gpt-4o';

    // Modern evaluation framework: Outcome (50%) + Process (30%) + Efficiency (20%)
    this.evaluationCriteria = {
      outcomeAchievement: {
        weight: 0.5,
        description: 'Goal completion and objective fulfillment'
      },
      processQuality: {
        weight: 0.3,
        description: 'Problem-solving approach, error recovery, and adaptability'
      },
      efficiency: {
        weight: 0.2,
        description: 'Time management, direct paths, and resource utilization'
      }
    };
  }

  private createSystemPrompt(meta: MetaData, prevSummary: string | null = null, isFinal: boolean = false): string {
    const basePrompt = `You are an advanced computer-use trajectory evaluator specialized in assessing human-computer interaction sequences. Your role is to provide nuanced, context-aware evaluation of user performance.

## TASK CONTEXT
Task ID: ${meta.id}
Title: ${meta.quest.title}
App: ${meta.quest.app}
User Request: ${meta.quest.content}

Objectives:
${meta.quest.objectives.map(objective => `- ${objective}`).join('\n')}

## EVALUATION FRAMEWORK
Use this modern holistic approach:

**Outcome Achievement (50%)**: Goal completion and objective fulfillment
- Did the user accomplish the primary objectives?
- How completely were the goals achieved?
- Were there any partial completions that show progress?

**Process Quality (30%)**: Problem-solving approach, error recovery, and adaptability
- How well did the user navigate obstacles?
- Did they recover effectively from errors?
- Was their approach logical and well-reasoned?
- Did they demonstrate creativity or resourcefulness?

**Efficiency (20%)**: Time management, direct paths, and resource utilization
- Were actions direct and purposeful?
- Did the user avoid unnecessary detours?
- How well did they manage their time and effort?

## EVALUATION PRIORITIES
**PRIMARY**: Observable actions and their direct outcomes
**SECONDARY**: Context clues from UI responses and system feedback
**IGNORE**: Unverified textual claims or assumptions
**FOCUS**: End-to-end goal accomplishment and problem-solving quality

## CHAIN-OF-THOUGHT PROCESS
For each evaluation, follow these steps:
1. **Identify the user's goal**: What were they trying to accomplish?
2. **Evaluate step-by-step execution**: How did they approach each objective?
3. **Assess obstacles and recovery**: How did they handle challenges?
4. **Judge efficiency and creativity**: Was their approach optimal?
5. **Calculate holistic score**: Combine all factors for final assessment

## CONFIDENCE ASSESSMENT
Rate your confidence (0.0-1.0) based on:
- **Action clarity**: How clear were the user's intentions?
- **Sequence completeness**: Did you see the full interaction?
- **Ambiguity factors**: Were there unclear or missing elements?`;

    if (prevSummary) {
      return basePrompt + `\n\n## PREVIOUS PROGRESS\n${prevSummary}\n\n${this.getChunkInstructions(isFinal)}`;
    }

    return basePrompt + `\n\n${this.getChunkInstructions(isFinal)}`;
  }

  private getChunkInstructions(isFinal: boolean): string {
    if (isFinal) {
      return `## FINAL EVALUATION
This is the final chunk. Provide a complete JSON evaluation following this exact format:

\`\`\`json
{
  "summary": "Comprehensive bullet-point overview of all progress made across all chunks",
  "analysis": "Detailed step-by-step reasoning following the chain-of-thought process",
  "score": 85,
  "reasoning": "Clear justification for the final score based on the evaluation framework",
  "confidence": 0.9,
  "outcomeAchievement": 80,
  "processQuality": 90,
  "efficiency": 85
}
\`\`\`

## EXAMPLE EVALUATION

**Task**: "Order a large pepperoni pizza for delivery"
**Objectives**: ["Navigate to pizza website", "Select large pepperoni pizza", "Add to cart", "Complete checkout with delivery"]

**Actions Observed**: User navigates to Domino's website, browses menu, adds large pepperoni to cart, but gets distracted and closes browser before checkout.

\`\`\`json
{
  "summary": "• Successfully navigated to pizza ordering website\n• Located and selected correct pizza size and toppings\n• Added item to cart successfully\n• Failed to complete checkout process - session abandoned",
  "analysis": "1. Goal Identification: User clearly understood the pizza ordering task\n2. Step-by-step Execution: Navigation and selection were executed well, showing familiarity with e-commerce interfaces\n3. Obstacles and Recovery: No significant obstacles encountered, but user failed to persist through checkout\n4. Efficiency Assessment: Initial actions were direct and purposeful\n5. Final Assessment: Strong start but critical failure to complete the primary goal",
  "score": 45,
  "reasoning": "While the user demonstrated competent navigation and selection skills (75% of objectives completed), the failure to complete checkout represents a critical gap in goal achievement. Outcome Achievement: 60% (3/4 objectives), Process Quality: 70% (good execution until abandonment), Efficiency: 80% (direct actions when engaged).",
  "confidence": 0.95,
  "outcomeAchievement": 60,
  "processQuality": 70,
  "efficiency": 80
}
\`\`\`

**Your response must be valid JSON only, no additional text.**`;
    } else {
      return `## CHUNK EVALUATION
Provide a JSON summary of progress combining previous summary (if any) with this chunk's accomplishments:

\`\`\`json
{
  "summary": "Comprehensive bullet-point overview of all progress made so far"
}
\`\`\`

**Your response must be valid JSON only, no additional text.**`;
    }
  }

  private chunkMessages(messages: Message[], chunkSize: number): Message[][] {
    // Filter out scroll messages first
    const filteredMessages = messages.filter(msg => {
      if (typeof msg.content === 'string') {
        let content = msg.content;
        // Remove python code block if present
        if (content.startsWith('```python\n')) {
          content = content.slice(10, -4); // Remove ```python\n and \n```
        }
        // Filter out if it starts with scroll
        return !content.startsWith('scroll(');
      }
      // Keep all image messages
      return true;
    });

    // Then chunk the filtered messages
    const chunks: Message[][] = [];
    for (let i = 0; i < filteredMessages.length; i += chunkSize) {
      chunks.push(filteredMessages.slice(i, i + chunkSize));
    }
    return chunks;
  }

  private extractClickCoordinates(message: string): [number, number] | null {
    const match = message.match(/click\((\d+),\s*(\d+)\)/);
    if (match) {
      return [parseInt(match[1]), parseInt(match[2])];
    }
    return null;
  }

  private formatMessageContent(content: string | { type: string; data: string }, prevMessage?: string): any {
    if (typeof content === 'string') {
      return content;
    }

    if (content.type === 'image') {
      let cropInfo = '';
      if (prevMessage) {
        const coords = this.extractClickCoordinates(prevMessage);
        if (coords) {
          cropInfo = ` The image is cropped to a 768x768 area centered around the cursor at coordinates ${coords}.`;
        }
      }

      return [
        {
          type: 'image_url',
          image_url: {
            url: `data:image/jpeg;base64,${content.data}`
          }
        },
        {
          type: 'text',
          text: `Screenshot of the application.${cropInfo}`
        }
      ];
    }

    return String(content);
  }

  private async evaluateChunk(
    systemPrompt: string,
    messages: Message[],
    isFinal: boolean,
    chunkIndex: number = 0,
    totalChunks: number = 1
  ): Promise<string | null> {
    try {
      // Add chunk metadata to system prompt
      const actionCount = messages.length;

      const enhancedSystemPrompt = `${systemPrompt}

## CHUNK ANALYSIS CONTEXT
**Chunk Progress**: ${chunkIndex + 1} of ${totalChunks} chunks
**Action Density**: ${actionCount} user actions in this segment
**Analysis Scope**: ${actionCount === 0 ? 'Static observation only' : 'Active user interaction sequence'}

## ADVANCED EVALUATION DIRECTIVES

### 🎯 **Action Boundary Recognition**
- **STRICT BOUNDARY**: Only evaluate actions between BEGIN_ACTIONS and END_ACTIONS markers
- **NO HALLUCINATION**: Never infer actions that aren't explicitly shown
- **SCREENSHOT CONTEXT**: Use visual information to understand action outcomes, not to assume actions

### 🧠 **Cognitive Load Assessment**
- **Task Complexity**: Consider the inherent difficulty of the current objectives
- **Context Switching**: Evaluate how well the user manages multiple UI elements
- **Decision Points**: Identify moments where the user had to make strategic choices

### 🔄 **Adaptive Evaluation Framework**
${actionCount === 0 ?
          `**ZERO-ACTION CHUNK PROTOCOL**:
- Focus on environmental observation and context preservation
- Note any UI changes or system responses that occurred
- Maintain continuity with previous progress without fabricating user actions
- Assess whether inaction was strategic (waiting) or problematic (confusion)` :
          `**ACTIVE CHUNK PROTOCOL**:
- Analyze action sequences for logical progression
- Evaluate decision quality at each interaction point
- Assess recovery strategies when encountering obstacles
- Measure efficiency through action-to-outcome ratios`}

### 📊 **Modern Scoring Considerations**
- **Outcome Achievement**: Did actions advance toward stated objectives?
- **Process Intelligence**: Was the approach methodical and well-reasoned?
- **Adaptive Behavior**: How well did the user respond to unexpected situations?
- **Efficiency Metrics**: Were actions direct and purposeful?

### 🔍 **Evidence-Based Analysis**
- **PRIMARY EVIDENCE**: Direct user actions (clicks, types, scrolls)
- **SECONDARY EVIDENCE**: UI responses, system feedback, visual changes
- **CONTEXTUAL CLUES**: Application state, error messages, success indicators
- **IGNORE**: Text overlays claiming completion without corresponding actions

${actionCount === 0 ?
          "⚠️ **CRITICAL**: This chunk contains no user actions. Focus on environmental continuity and context preservation. Do not fabricate user interactions." :
          `✅ **ACTIVE ANALYSIS**: ${actionCount} actions detected. Evaluate the complete interaction sequence for strategic coherence and goal progression.`}`;

      const formattedMessages: Array<{
        role: 'system' | 'user' | 'assistant';
        content: any;
      }> = [{ role: 'system', content: enhancedSystemPrompt }];

      // Add a clear marker for the beginning of actions with context
      const contextualIntro = actionCount === 0
        ? `=== BEGIN_OBSERVATION_WINDOW ===
📸 STATIC ANALYSIS: This chunk contains screenshots only - no user actions to evaluate.
🔍 FOCUS: Environmental context, UI state, and continuity with previous progress.
⚠️  DO NOT infer or hallucinate user actions that aren't explicitly documented.`
        : `=== BEGIN_ACTIONS (${actionCount} user interactions) ===
🎯 ACTIVE ANALYSIS: Evaluating ${actionCount} user action${actionCount > 1 ? 's' : ''} in sequence.
🧠 FOCUS: Action logic, decision quality, goal progression, and efficiency.
📊 CONTEXT: Chunk ${chunkIndex + 1}/${totalChunks} - ${Math.round((actionCount / Math.max(1, totalChunks)) * 100)}% action density.`;

      formattedMessages.push({
        role: 'user',
        content: contextualIntro
      });

      for (let i = 0; i < messages.length; i++) {
        const prevMessage = i > 0 ? messages[i - 1].content : undefined;
        formattedMessages.push({
          role: 'user',
          content: this.formatMessageContent(messages[i].content, typeof prevMessage === 'string' ? prevMessage : undefined)
        });
      }

      // Add a clear marker for the end of actions with summary
      const contextualOutro = actionCount === 0
        ? `=== END_OBSERVATION_WINDOW ===
📋 SUMMARY: Static analysis complete - no user actions detected.
🎯 NEXT: Provide environmental context and continuity assessment in JSON format.`
        : `=== END_ACTIONS (${actionCount} user interactions) ===
📋 SEQUENCE COMPLETE: All ${actionCount} user action${actionCount > 1 ? 's' : ''} documented above.
🎯 NEXT: Provide comprehensive evaluation following the modern framework in JSON format.
⚡ REMINDER: Focus on outcome achievement, process quality, and efficiency.`;

      formattedMessages.push({
        role: 'user',
        content: contextualOutro
      });

      const response = await this.client.chat.completions.create({
        model: this.model,
        messages: formattedMessages,
        max_tokens: 2000,
        temperature: 0,
        response_format: { type: "json_object" }
      });

      return response.choices[0].message.content;
    } catch (error) {
      console.error('Error calling OpenAI API:', error);
      return null;
    }
  }

  private parseJsonResponse(text: string): any {
    try {
      // Extract JSON from code blocks if present
      const jsonMatch = text.match(/```json\s*({[\s\S]*?})\s*```/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[1]);
      }

      // Try parsing the entire response as JSON
      return JSON.parse(text.trim());
    } catch (error) {
      console.error('Failed to parse JSON response:', error);
      console.error('Raw response:', text);
      return null;
    }
  }

  async grade(metaPath: string, sftPath: string): Promise<GradeResult | null> {
    try {
      // Read input files
      const meta: MetaData = await Bun.file(metaPath).json();
      const sft = await Bun.file(sftPath).json();

      // Split messages into chunks
      const chunks = this.chunkMessages(sft, this.chunkSize);
      const totalChunks = chunks.length;

      console.log(`Processing ${totalChunks} chunks...`);

      // Process each chunk
      let prevSummary: string | null = null;
      for (let i = 0; i < chunks.length; i++) {
        const isFinal = i === chunks.length - 1;
        const chunk = chunks[i];

        console.log(`\nProcessing chunk ${i + 1}/${totalChunks}`);
        const systemPrompt = this.createSystemPrompt(meta, prevSummary, isFinal);
        const evaluation = await this.evaluateChunk(systemPrompt, chunk, isFinal, i, totalChunks);

        if (!evaluation) {
          console.log('Failed to get evaluation');
          continue;
        }

        if (isFinal) {
          const finalEval = this.parseJsonResponse(evaluation) as FinalEvaluation;

          if (!finalEval || !finalEval.summary || typeof finalEval.score !== 'number' || !finalEval.reasoning) {
            console.log('Failed to parse final evaluation JSON, retrying chunk...');
            i--; // Retry this chunk
            continue;
          }

          // Validate score components
          const outcomeScore = finalEval.outcomeAchievement || 0;
          const processScore = finalEval.processQuality || 0;
          const efficiencyScore = finalEval.efficiency || 0;
          const confidence = Math.max(0, Math.min(1, finalEval.confidence || 0.5));

          console.log({
            summary: finalEval.summary,
            analysis: finalEval.analysis,
            score: finalEval.score,
            reasoning: finalEval.reasoning,
            confidence: confidence,
            breakdown: {
              outcome: outcomeScore,
              process: processScore,
              efficiency: efficiencyScore
            }
          });

          return {
            summary: finalEval.summary,
            analysis: finalEval.analysis || finalEval.reasoning,
            score: finalEval.score,
            reasoning: finalEval.reasoning,
            confidence: confidence,
            outcomeAchievement: outcomeScore,
            processQuality: processScore,
            efficiency: efficiencyScore
          };
        } else {
          const chunkEval = this.parseJsonResponse(evaluation) as ChunkEvaluation;

          if (!chunkEval || !chunkEval.summary) {
            console.log('Failed to parse chunk evaluation JSON, retrying chunk...');
            i--; // Retry this chunk
            continue;
          }

          prevSummary = chunkEval.summary;
          console.log(`Progress Summary: ${prevSummary}`);
        }
      }

      return null;
    } catch (error) {
      console.error('Error during grading:', error);
      return null;
    }
  }
}
