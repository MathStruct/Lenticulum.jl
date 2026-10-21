import { siteConfig } from "./mathstruct/config"

/**
 * Quartz configuration for the Lenticulum theory vault. The plugins, theme and fixes shared by all
 * MathStruct sites are in the preset (https://github.com/MathStruct/quartz,
 * mathstruct/config.ts); this file only says what is particular to this vault.
 *
 * docs/site/build.sh stages the vault and builds it with this file; see docs/site/README.md.
 * Deployed as the front page of the site (docs/site/deploy_vault.sh); the Documenter site is
 * under api/.
 */
export default siteConfig({
  pageTitle: "Lenticulum — theory vault",
  pageTitleSuffix: " · Lenticulum",
  locale: "en-GB",
  // A preview build (docs/make.jl off `master`) replaces this with …/preview. The custom
  // domain, not mathstruct.github.io: that redirects here, and giscus loads its theme
  // stylesheet from this URL, which a redirect (without a CORS header) would block.
  baseUrl: process.env.VAULT_BASEURL || "mathstruct.org/Lenticulum.jl",
  // Prompts/ holds the authoring prompts, not vault content
  ignorePatterns: ["templates", "Prompts"],
  // build.sh sets each staged file's date from git (git_dates.py), so the filesystem date is
  // the source's last change
  dateSources: ["frontmatter", "filesystem"],
  rss: false,
})
