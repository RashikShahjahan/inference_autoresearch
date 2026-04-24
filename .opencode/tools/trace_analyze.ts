import { tool } from "@opencode-ai/plugin"

import { runPythonTool } from "./_shared"

export default tool({
  description: "Analyze an Instruments Metal trace for inference bottlenecks",
  args: {
    trace_path: tool.schema
      .string()
      .default("state/batch_generate_profile.trace")
      .describe("Path to a .trace document, relative to repo root or absolute"),
    run_number: tool.schema
      .number()
      .int()
      .positive()
      .default(1)
      .describe("Trace run number to analyze"),
    process_name: tool.schema
      .string()
      .default("python3")
      .describe("Process name filter for inference rows"),
    cluster_gap_ms: tool.schema
      .number()
      .positive()
      .default(500)
      .describe("Start-to-start gap threshold used to split warmup and measured clusters"),
    top_n: tool.schema
      .number()
      .int()
      .positive()
      .default(15)
      .describe("Number of top grouped operations to report"),
  },
  async execute(args, context) {
    return runPythonTool(
      "trace_analyze.py",
      [
        "--trace-path",
        args.trace_path,
        "--run-number",
        String(args.run_number),
        "--process-name",
        args.process_name,
        "--cluster-gap-ms",
        String(args.cluster_gap_ms),
        "--top-n",
        String(args.top_n),
      ],
      {},
      context.worktree,
    )
  },
})
