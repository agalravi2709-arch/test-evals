// apps/server/src/index.ts

import { auth } from "@test-evals/auth";
import { env } from "@test-evals/env/server";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { logger } from "hono/logger";

import fs from "fs";
import path from "path";
import pLimit from "p-limit";
import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";
import stringSimilarity from "string-similarity";

/* =========================
   🧠 SCHEMA
========================= */
const Schema = z.object({
  chief_complaint: z.string(),
  vitals: z.object({
    bp: z.string().nullable(),
    hr: z.number().nullable(),
    temp_f: z.number().nullable(),
    spo2: z.number().nullable(),
  }),
  medications: z.array(
    z.object({
      name: z.string(),
      dose: z.string(),
      frequency: z.string(),
      route: z.string(),
    })
  ),
  diagnoses: z.array(
    z.object({
      description: z.string(),
      icd10: z.string().optional(),
    })
  ),
  plan: z.array(z.string()),
  follow_up: z.object({
    interval_days: z.number().nullable(),
    reason: z.string().nullable(),
  }),
});

/* =========================
   🤖 LLM SETUP
========================= */
const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY!,
});

function buildPrompt(strategy: string) {
  const base = `
You extract structured clinical data.
Return ONLY valid JSON.
`;

  if (strategy === "few_shot") {
    return base + `Example: patient has fever → {"chief_complaint":"fever"}`;
  }

  if (strategy === "cot") {
    return base + `Think step by step internally.`;
  }

  return base;
}

/* =========================
   🔁 EXTRACTION + RETRY
========================= */
async function extractWithRetry(transcript: string, prompt: string) {
  let errorMsg = "";

  for (let i = 1; i <= 3; i++) {
    const res = await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 1200,
      messages: [
        {
          role: "user",
          content: `${prompt}

Transcript:
${transcript}

Error:
${errorMsg}`,
        },
      ],
      tools: [
        {
          name: "extract",
          input_schema: Schema,
        },
      ],
    });

    const tool = res.content.find((c: any) => c.type === "tool_use");

    if (!tool) continue;

    try {
      const parsed = Schema.parse(tool.input);
      return parsed;
    } catch (e: any) {
      errorMsg = e.message;
    }
  }

  throw new Error("Failed after retries");
}

/* =========================
   📊 EVALUATION
========================= */
function fuzzy(a: string, b: string) {
  return stringSimilarity.compareTwoStrings(
    a.toLowerCase(),
    b.toLowerCase()
  );
}

function setF1(pred: string[], gold: string[]) {
  let match = 0;

  for (const p of pred) {
    if (gold.some((g) => fuzzy(p, g) > 0.8)) match++;
  }

  const precision = match / (pred.length || 1);
  const recall = match / (gold.length || 1);

  if (!precision && !recall) return 0;

  return (2 * precision * recall) / (precision + recall);
}

function evaluate(pred: any, gold: any) {
  return {
    chief_complaint: fuzzy(
      pred.chief_complaint,
      gold.chief_complaint
    ),

    medications: setF1(
      pred.medications.map((m: any) => m.name),
      gold.medications.map((m: any) => m.name)
    ),

    diagnoses: setF1(
      pred.diagnoses.map((d: any) => d.description),
      gold.diagnoses.map((d: any) => d.description)
    ),

    plan: setF1(pred.plan, gold.plan),
  };
}

/* =========================
   🚀 RUNNER
========================= */
const limit = pLimit(5);

async function runEval(strategy: string) {
  const transcriptsDir = path.join(process.cwd(), "data/transcripts");
  const goldDir = path.join(process.cwd(), "data/gold");

  const files = fs.readdirSync(transcriptsDir);

  const prompt = buildPrompt(strategy);

  const results: any[] = [];

  await Promise.all(
    files.map((file) =>
      limit(async () => {
        const transcript = fs.readFileSync(
          path.join(transcriptsDir, file),
          "utf-8"
        );

        const gold = JSON.parse(
          fs.readFileSync(
            path.join(goldDir, file.replace(".txt", ".json")),
            "utf-8"
          )
        );

        const pred = await extractWithRetry(
          transcript,
          prompt
        );

        const score = evaluate(pred, gold);

        results.push({ file, score });
      })
    )
  );

  return results;
}

/* =========================
   🌐 SERVER
========================= */
const app = new Hono();

app.use(logger());
app.use(
  "/*",
  cors({
    origin: env.CORS_ORIGIN,
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
    credentials: true,
  }),
);

app.on(["POST", "GET"], "/api/auth/*", (c) =>
  auth.handler(c.req.raw)
);

app.get("/", (c) => c.text("OK"));

app.post("/api/v1/runs", async (c) => {
  const { strategy = "zero_shot" } = await c.req.json();

  const result = await runEval(strategy);

  return c.json({ result });
});

export default app;
app.on(["POST", "GET"], "/api/auth/*", (c) => auth.handler(c.req.raw));

app.get("/", (c) => {
  return c.text("OK");
});

export default app;
