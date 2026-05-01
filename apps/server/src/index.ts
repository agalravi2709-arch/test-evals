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
   🤖 LLM
========================= */
const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY!,
});

function buildPrompt(strategy: string) {
  const base = `
You extract structured clinical data.
Return ONLY valid JSON matching schema.
`;

  if (strategy === "few_shot") {
    return base + `
Example:
Transcript: patient has fever 101F
Output: {"chief_complaint":"fever"}
`;
  }

  if (strategy === "cot") {
    return base + `Think step by step internally before output.`;
  }

  return base;
}

/* =========================
   🔁 EXTRACTION + RETRY
========================= */
async function extractWithRetry(transcript: string, prompt: string) {
  let errorMsg = "";
  let lastError = null;

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

Previous error:
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
      return { parsed, attempts: i, failed: false };
    } catch (e: any) {
      errorMsg = e.message;
      lastError = e.message;
    }
  }

  return { parsed: null, attempts: 3, failed: true, error: lastError };
}

/* =========================
   📊 METRICS
========================= */
function fuzzy(a: string, b: string) {
  return stringSimilarity.compareTwoStrings(
    (a || "").toLowerCase(),
    (b || "").toLowerCase()
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

/* =========================
   🚨 HALLUCINATION DETECTOR
========================= */
function isGrounded(value: any, transcript: string) {
  if (!value) return true;

  const val = String(value).toLowerCase();
  return transcript.toLowerCase().includes(val);
}

function countHallucinations(pred: any, transcript: string) {
  let count = 0;

  if (!isGrounded(pred.chief_complaint, transcript)) count++;

  pred.plan.forEach((p: string) => {
    if (!isGrounded(p, transcript)) count++;
  });

  pred.medications.forEach((m: any) => {
    if (!isGrounded(m.name, transcript)) count++;
  });

  pred.diagnoses.forEach((d: any) => {
    if (!isGrounded(d.description, transcript)) count++;
  });

  return count;
}

/* =========================
   📊 EVALUATOR
========================= */
function evalVitals(pred: any, gold: any) {
  const keys = ["bp", "hr", "temp_f", "spo2"];
  let score = 0;

  keys.forEach((k) => {
    const p = pred[k];
    const g = gold[k];

    if (p == null && g == null) score += 1;
    else if (typeof g === "number") {
      if (Math.abs(p - g) <= 0.2) score += 1;
    } else if (p === g) {
      score += 1;
    }
  });

  return score / keys.length;
}

function evaluate(pred: any, gold: any) {
  return {
    chief_complaint: fuzzy(
      pred.chief_complaint,
      gold.chief_complaint
    ),

    vitals: evalVitals(pred.vitals, gold.vitals),

    medications: setF1(
      pred.medications.map((m: any) => m.name),
      gold.medications.map((m: any) => m.name)
    ),

    diagnoses: setF1(
      pred.diagnoses.map((d: any) => d.description),
      gold.diagnoses.map((d: any) => d.description)
    ),

    plan: setF1(pred.plan, gold.plan),

    follow_up:
      pred.follow_up.interval_days ===
      gold.follow_up.interval_days
        ? 1
        : fuzzy(
            pred.follow_up.reason || "",
            gold.follow_up.reason || ""
          ),
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

  let schemaFailures = 0;
  let hallucinations = 0;

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

        const res = await extractWithRetry(
          transcript,
          prompt
        );

        if (res.failed || !res.parsed) {
          schemaFailures++;
          return;
        }

        const pred = res.parsed;

        const score = evaluate(pred, gold);

        const hallucinationCount = countHallucinations(
          pred,
          transcript
        );

        hallucinations += hallucinationCount;

        results.push({
          file,
          score,
          attempts: res.attempts,
          hallucinations: hallucinationCount,
        });
      })
    )
  );

  return {
    total_cases: files.length,
    completed: results.length,
    schema_failures: schemaFailures,
    hallucinations,
    results,
  };
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

  return c.json(result);
});

export default app;
