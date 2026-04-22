import { tool } from "@opencode-ai/plugin"

import { runPythonTool } from "./_shared"

export default tool({
  description: "Capture a representative Metal GPU trace",
  args: {
    trace_path: tool.schema
      .string()
      .default("state/batch_generate_profile.gputrace")
      .describe("Output .gputrace path relative to the repo root or absolute"),
    fixture_count: tool.schema
      .number()
      .int()
      .positive()
      .default(1)
      .describe("Number of fixtures to include in the representative batch"),
  },
  async execute(args, context) {
    return runPythonTool(
      "capture_gpu_trace.py",
      [
        "--metal-profile-path",
        args.trace_path,
        "--metal-profile-fixture-count",
        String(args.fixture_count),
      ],
      { MTL_CAPTURE_ENABLED: "1" },
      context.worktree,
    )
  },
})
