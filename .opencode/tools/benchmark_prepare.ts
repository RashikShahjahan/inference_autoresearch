import { tool } from "@opencode-ai/plugin"

import { runPythonTool } from "./_shared"

export default tool({
  description: "Initialize or validate benchmark state",
  args: {
    overwrite_incumbent: tool.schema
      .boolean()
      .default(false)
      .describe("Replace state/best_generate.py with the current generate.py even if it already exists"),
  },
  async execute(args, context) {
    const toolArgs = args.overwrite_incumbent ? ["--overwrite-incumbent"] : []
    return runPythonTool("benchmark_prepare.py", toolArgs, {}, context.worktree)
  },
})
