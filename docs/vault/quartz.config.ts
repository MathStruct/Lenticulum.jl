import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz configuration for the Lenticulum theory vault.
 *
 * This file is copied over the stock `quartz.config.ts` of a Quartz v4 checkout by
 * `docs/vault/build.sh`. The vault is staged into `content/` by the same script; see
 * `docs/vault/README.md` for the layout.
 *
 * Deployed beside the Documenter site, at `<documenter-root>/vault/`.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "Lenticulum — theory vault",
    pageTitleSuffix: " · Lenticulum",
    enableSPA: true,
    enablePopovers: true,
    analytics: null,
    locale: "en-GB",
    // The Documenter site deploys to <root>/dev/ from `master`; the vault lives under it.
    baseUrl: "mathstruct.github.io/Lenticulum.jl/dev/vault",
    // Prompts/ holds the authoring prompts, not vault content.
    ignorePatterns: ["private", "templates", ".obsidian", "Prompts", ".tikz-cache"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Schibsted Grotesk",
        body: "Source Sans Pro",
        code: "IBM Plex Mono",
      },
      colors: {
        lightMode: {
          light: "#faf8f8",
          lightgray: "#e5e5e5",
          gray: "#b8b8b8",
          darkgray: "#4e4e4e",
          dark: "#2b2b2b",
          secondary: "#284b63",
          tertiary: "#84a59d",
          highlight: "rgba(143, 159, 169, 0.15)",
          textHighlight: "#fff23688",
        },
        darkMode: {
          light: "#161618",
          lightgray: "#393639",
          gray: "#646464",
          darkgray: "#d4d4d4",
          dark: "#ebebec",
          secondary: "#7b97aa",
          tertiary: "#84a59d",
          highlight: "rgba(143, 159, 169, 0.15)",
          textHighlight: "#b3aa0288",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({ priority: ["frontmatter", "filesystem"] }),
      Plugin.SyntaxHighlighting({
        theme: { light: "github-light", dark: "github-dark" },
        keepBackground: false,
      }),
      // ```tikz blocks → inline SVG at build time (node-tikzjax), cached by content hash in
      // docs/vault/.tikz-cache/, which is committed so CI only renders new diagrams.
      Plugin.TikZ({ cacheDir: ".tikz-cache" }),
      // Obsidian syntax: [[wikilinks]], > [!callouts], etc.
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      // "shortest" resolves [[messages]] by basename, exactly as Obsidian does.
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex", katexOptions: { throwOnError: false, strict: false } }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({ enableSiteMap: true, enableRSS: false }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      // CustomOgImages is slow and needs an absolute base; not worth it here.
    ],
  },
}

export default config
