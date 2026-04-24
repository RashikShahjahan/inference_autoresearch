import { tool } from "@opencode-ai/plugin"

import { runPythonTool } from "./_shared"

export default tool({
  description: "Capture a representative Metal System Trace",
  args: {
    trace_path: tool.schema
      .string()
      .default("state/batch_generate_profile.trace")
      .describe("Output .trace path relative to the repo root or absolute"),
    fixture_count: tool.schema
      .number()
      .int()
      .positive()
      .default(16)
      .describe("Number of fixtures to include in the representative batch"),
  },
  async execute(args, context) {
    return runPythonTool(
      "capture_trace.py",
      ["--trace-path", args.trace_path, "--fixture-count", String(args.fixture_count)],
      {},
      context.worktree,
    )
  },
})
