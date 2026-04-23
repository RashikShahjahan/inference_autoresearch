import { tool } from "@opencode-ai/plugin"

import { runPythonTool } from "./_shared"

export default tool({
  description: "Export a trace table of contents",
  args: {
    trace_path: tool.schema
      .string()
      .default("state/batch_generate_profile.trace")
      .describe("Path to a .trace document, relative to repo root or absolute"),
  },
  async execute(args, context) {
    return runPythonTool("trace_toc.py", ["--trace-path", args.trace_path], {}, context.worktree)
  },
})
