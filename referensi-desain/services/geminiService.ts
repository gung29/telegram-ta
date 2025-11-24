import { GoogleGenAI, Type } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export interface AnalysisResult {
  score: number;
  category: string;
  reasoning: string;
}

export const analyzeTextWithGemini = async (text: string): Promise<AnalysisResult> => {
  try {
    const model = "gemini-2.5-flash";
    const response = await ai.models.generateContent({
      model: model,
      contents: `Analyze the following text for hate speech, toxicity, or harmful content.
      Text: "${text}"
      
      Return a JSON object with:
      - score: A number between 0 and 100 representing the likelihood of hate speech (100 being certain hate speech).
      - category: A short string classification (e.g., "Hate Speech", "Harassment", "Spam", "Safe", "Controversial").
      - reasoning: A brief explanation (max 1 sentence).`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            score: { type: Type.NUMBER },
            category: { type: Type.STRING },
            reasoning: { type: Type.STRING },
          },
          required: ["score", "category", "reasoning"],
        }
      }
    });

    const result = JSON.parse(response.text || "{}");
    return {
      score: result.score || 0,
      category: result.category || "Unknown",
      reasoning: result.reasoning || "No reasoning provided."
    };
  } catch (error) {
    console.error("Gemini Analysis Error:", error);
    return {
      score: 0,
      category: "Error",
      reasoning: "Failed to analyze text."
    };
  }
};