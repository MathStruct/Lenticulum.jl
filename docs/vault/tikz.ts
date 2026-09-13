// Ported from MathStruct/CategoryTheory-ML-Wiki (site/quartz/plugins/transformers/tikz.ts).
// Copied into the Quartz checkout by `docs/vault/build.sh` and registered in
// `quartz/plugins/transformers/index.ts` there.
//
import { QuartzTransformerPlugin } from "../types"
import { Root, Code } from "mdast"
import { visit } from "unist-util-visit"
import { createHash } from "crypto"
import fs from "fs"
import path from "path"
import * as tikzjax from "node-tikzjax"
// CommonJS interop: the default export may be nested under `.default`
const tex2svg: (src: string, o?: any) => Promise<string> =
  (tikzjax as any).default?.default ?? (tikzjax as any).default ?? (tikzjax as any)

/**
 * Renders ```tikz code blocks (Obsidian "inline-tikz"/"tikzjax" format:
 * \usepackage lines, then \begin{document} ... \end{document}) to inline SVG
 * at build time with node-tikzjax. Results are cached on disk by content hash.
 */
interface Options {
  cacheDir: string
  fontCssUrl: string
}

const defaultOptions: Options = {
  cacheDir: ".tikz-cache",
  fontCssUrl: "https://cdn.jsdelivr.net/npm/node-tikzjax@1.0.5/css/fonts.css",
}

// node-tikzjax must not run concurrently within one process: serialize renders.
let queue: Promise<unknown> = Promise.resolve()
function serialize<T>(task: () => Promise<T>): Promise<T> {
  const next = queue.then(task, task)
  queue = next.catch(() => {})
  return next
}

function splitSource(src: string): { preamble: string; body: string } {
  const lines = src.split("\n")
  const preamble: string[] = []
  const body: string[] = []
  let inDoc = false
  for (const line of lines) {
    if (!inDoc && !line.includes("\\begin{document}")) {
      preamble.push(line)
    } else {
      inDoc = true
      body.push(line)
    }
  }
  if (!inDoc) {
    // bare tikzpicture without \begin{document}
    return { preamble: preamble.filter((l) => l.trim().startsWith("\\use")).join("\n"),
             body: `\\begin{document}\n${src}\n\\end{document}` }
  }
  return { preamble: preamble.join("\n"), body: body.join("\n") }
}

async function render(src: string, opts: Options): Promise<string> {
  const hash = createHash("sha256").update(src).digest("hex")
  const cachePath = path.join(opts.cacheDir, `${hash}.svg`)
  if (fs.existsSync(cachePath)) return fs.readFileSync(cachePath, "utf8")
  const { preamble, body } = splitSource(src)
  const svg = await serialize(() =>
    tex2svg(body, {
      showConsole: false,
      texPackages: { amssymb: "", amstext: "" },
      addToPreamble: preamble,
      embedFontCss: false,
      fontCssUrl: opts.fontCssUrl,
    }),
  )
  fs.mkdirSync(opts.cacheDir, { recursive: true })
  fs.writeFileSync(cachePath, svg)
  return svg
}

export const TikZ: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: "TikZ",
    markdownPlugins() {
      return [
        () => async (tree: Root, file) => {
          const jobs: Array<Promise<void>> = []
          visit(tree, "code", (node: Code, index, parent) => {
            if (node.lang !== "tikz" || !parent || index === undefined) return
            const job = render(node.value, opts)
              .then((svg) => {
                parent.children[index] = {
                  type: "html",
                  value: `<figure class="tikz">${svg}</figure>`,
                } as any
              })
              .catch((err) => {
                console.warn(`\nTikZ render failed in ${file.path}: ${String(err).slice(0, 200)}`)
                parent.children[index] = {
                  type: "html",
                  value: `<pre class="tikz-error"><code>${node.value
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")}</code></pre>`,
                } as any
              })
            jobs.push(job)
          })
          await Promise.all(jobs)
        },
      ]
    },
    externalResources() {
      return { css: [{ content: opts.fontCssUrl }] }
    },
  }
}
