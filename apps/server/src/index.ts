// apps/server/src/index.ts
import { auth } from "@test-evals/auth";
import { env } from "@test-evals/env/server";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { logger } from "hono/logger";

// ✅ ADD THIS
import { runEval } from "./services/runner.service";

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

app.on(["POST", "GET"], "/api/auth/*", (c) => auth.handler(c.req.raw));

app.get("/", (c) => {
  return c.text("OK");
});


// ✅ ADD THIS ROUTE
app.post("/api/v1/runs", async (c) => {
  try {
    const body = await c.req.json();
    const { strategy = "zero_shot" } = body;

    const result = await runEval(strategy);

    return c.json({
      status: "success",
      result,
    });
  } catch (err: any) {
    return c.json(
      {
        status: "error",
        message: err.message,
      },
      500
    );
  }
});

export default app;
