import path from "path"

export async function runPythonTool(
  scriptName: string,
  args: string[],
  env: Record<string, string>,
  cwd: string,
) {
  const scriptPath = path.join(cwd, ".opencode/tools", scriptName)
  const proc = Bun.spawn(["uv", "run", "python3", scriptPath, ...args], {
    cwd,
    env: { ...process.env, ...env },
    stdout: "pipe",
    stderr: "pipe",
  })

  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ])

  const trimmedStdout = stdout.trim()
  const trimmedStderr = stderr.trim()

  if (!trimmedStdout) {
    throw new Error(trimmedStderr || `${scriptName} exited with code ${exitCode}`)
  }

  let result: unknown
  try {
    result = JSON.parse(trimmedStdout)
  } catch (error) {
    throw new Error(
      `Failed to parse ${scriptName} output as JSON: ${error instanceof Error ? error.message : String(error)}\n${trimmedStdout}`,
    )
  }

  if (exitCode !== 0) {
    throw new Error(trimmedStderr || `${scriptName} exited with code ${exitCode}`)
  }

  return result
}
