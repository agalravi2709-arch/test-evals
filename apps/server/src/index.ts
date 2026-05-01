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
import crypto from "crypto";

/* =========================
   CLI MODE
========================= */
const isCLI =
  process.argv.includes("--cli") || process.argv.includes("eval");

/* =========================
   SCHEMA
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
   ANTHROPIC CLIENT
========================= */
const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY!,
});

/* =========================
   JSON SCHEMA FIX (IMPORTANT)
========================= */
function toJSONSchema() {
  return {
    type: "object",
    properties: {
      chief_complaint: { type: "string" },
      vitals: {
        type: "object",
        properties: {
          bp: { type: ["string", "null"] },
          hr: { type: ["number", "null"] },
          temp_f: { type: ["number", "null"] },
          spo2: { type: ["number", "null"] },
        },
      },
      medications: {
        type: "array",
        items: {
          type: "object",
          properties: {
            name: { type: "string" },
            dose: { type: "string" },
            frequency: { type: "string" },
            route: { type: "string" },
          },
        },
      },
      diagnoses: { type: "array", items: { type: "object" } },
      plan: { type: "array", items: { type: "string" } },
      follow_up: { type: "object" },
    },
  };
}

/* =========================
   PROMPTS
========================= */
function buildPrompt(strategy: string) {
  const base = `Extract structured clinical JSON. Return ONLY valid JSON.`;

  if (strategy === "few_shot") {
    return base + ` Example: fever → {"chief_complaint":"fever"}`;
  }

  if (strategy === "cot") {
    return base + ` Think step-by-step internally.`;
  }

  return base;
}

/* =========================
   RETRY + SAFE EXTRACTION
========================= */
async function extractWithRetry(transcript: string, prompt: string) {
  let errorMsg = "";

  for (let i = 1; i <= 3; i++) {
    try {
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
            input_schema: toJSONSchema(),
          },
        ],
      });

      const tool = res.content.find((c: any) => c.type === "tool_use");

      if (!tool) {
        errorMsg = "NO_TOOL_OUTPUT";
        continue;
      }

      const parsed = Schema.parse(tool.input);
      return { parsed, attempts: i, failed: false };
    } catch (err: any) {
      errorMsg = err.message;
    }
  }

  return { parsed: null, attempts: 3, failed: true };
}

/* =========================
   METRICS
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

  return precision + recall === 0
    ? 0
    : (2 * precision * recall) / (precision + recall);
}

/* =========================
   HALLUCINATION CHECK
========================= */
function grounded(val: string, transcript: string) {
  return transcript.toLowerCase().includes((val || "").toLowerCase());
}

function hallucinationScore(pred: any, transcript: string) {
  let count = 0;

  if (!grounded(pred.chief_complaint, transcript)) count++;

  pred.plan.forEach((p: string) => {
    if (!grounded(p, transcript)) count++;
  });

  pred.medications.forEach((m: any) => {
    if (!grounded(m.name, transcript)) count++;
  });

  return count;
}

/* =========================
   EVALUATION
========================= */
function evaluate(pred: any, gold: any) {
  return {
    chief_complaint: fuzzy(pred.chief_complaint, gold.chief_complaint),

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
      pred.follow_up?.interval_days === gold.follow_up?.interval_days
        ? 1
        : fuzzy(pred.follow_up?.reason || "", gold.follow_up?.reason || ""),
  };
}

/* =========================
   RUNNER
========================= */
const limit = pLimit(5);

async function runEval(strategy: string) {
  const files = fs.readdirSync(
    path.join(process.cwd(), "data/transcripts")
  );

  const prompt = buildPrompt(strategy);
  const runId = crypto.createHash("sha256").update(prompt).digest("hex");

  const results: any[] = [];
  let schemaFailures = 0;
  let hallucinations = 0;

  const start = Date.now();

  await Promise.all(
    files.map((file) =>
      limit(async () => {
        const transcript = fs.readFileSync(
          path.join(process.cwd(), "data/transcripts", file),
          "utf-8"
        );

        const gold = JSON.parse(
          fs.readFileSync(
            path.join(process.cwd(), "data/gold", file.replace(".txt", ".json")),
            "utf-8"
          )
        );

        const res = await extractWithRetry(transcript, prompt);

        if (res.failed || !res.parsed) {
          schemaFailures++;
          return;
        }

        const pred = res.parsed;

        const score = evaluate(pred, gold);

        const h = hallucinationScore(pred, transcript);
        hallucinations += h;

        results.push({
          file,
          score,
          hallucinations: h,
          attempts: res.attempts,
        });
      })
    )
  );

  const duration = (Date.now() - start) / 1000;

  return {
    runId,
    strategy,
    summary: {
      total: files.length,
      completed: results.length,
      schemaFailures,
      hallucinations,
      duration,
    },
    results,
  };
}

/* =========================
   CLI OUTPUT
========================= */
async function main() {
  const strategy =
    process.argv.find((a) => a.startsWith("--strategy="))?.split("=")[1] ||
    "zero_shot";

  console.log("🚀 Running eval:", strategy);

  const result = await runEval(strategy);

  const avg =
    result.results.reduce((acc, r) => {
      const s = r.score;
      return (
        acc +
        (s.chief_complaint +
          s.medications +
          s.diagnoses +
          s.plan +
          s.follow_up) /
          5
      );
    }, 0) / (result.results.length || 1);

  console.log("\n====================");
  console.log("📊 SUMMARY");
  console.log("====================");
  console.log("Run ID:", result.runId);
  console.log("Strategy:", result.strategy);
  console.log("Total:", result.summary.total);
  console.log("Completed:", result.summary.completed);
  console.log("Schema failures:", result.summary.schemaFailures);
  console.log("Hallucinations:", result.summary.hallucinations);
  console.log("Duration (s):", result.summary.duration.toFixed(2));
  console.log("Avg score:", avg.toFixed(3));

  process.exit(0);
}

/* =========================
   SERVER
========================= */
const app = new Hono();

app.use(logger());
app.use(
  "/*",
  cors({
    origin: env.CORS_ORIGIN,
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type"],
  })
);

app.get("/", (c) => c.text("OK"));

app.post("/api/v1/runs", async (c) => {
  const { strategy = "zero_shot" } = await c.req.json();

  if (!["zero_shot", "few_shot", "cot"].includes(strategy)) {
    return c.json({ error: "Invalid strategy" }, 400);
  }

  const result = await runEval(strategy);
  return c.json(result);
});

export default isCLI ? main : app;
